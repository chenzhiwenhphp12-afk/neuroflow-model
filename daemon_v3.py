"""
NeuroFlow 自主进化守护进程 v3 — 内置事件循环 + 混合数据源
==========================================================
零外部依赖：不依赖 Cron，不依赖外部 API（网络仅作可选的增强）
架构：内置调度器 → 本地知识缓存(优先) → 网络API(可选增强) → 自我进化

改进 vs v2/daemon.py:
  1. 完全内置事件循环（while True + 自适应 sleep），不再依赖外部 Cron
  2. 混合数据源：本地知识库优先，网络 API 作为可选增强
  3. 本地 KB 包含 595 个已学知识文件 + 内置 200+ 跨领域知识点
  4. TrainableHead SGD 替代 SelfEvolution 噪声扰动
  5. 状态自检 + 本地文件报告

用法: nohup python3 daemon_v3.py &> daemon_v3.log &
      或: python3 daemon_v3.py  (前台运行)
"""

import sys, os, time, json, random, logging, numpy as np
from datetime import datetime
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import urllib.request
import urllib.error

# ── 路径 ──────────────────────────────────────
sys.path.insert(0, "/mnt/d/neuroflow-model")

# ── 配置 ──────────────────────────────────────
DEPLOY_PATH = "/mnt/d/neuroflow-model"
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_state.json")
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights_v4.npz"
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")
VOCAB_FILE = os.path.join(DEPLOY_PATH, "char_vocab.json")
MASK_RATIO = 0.35           # 掩码比例（35% — 增大难度提升表征质量）
INPUT_NOISE = 0.05          # 输入高斯噪声标准差
REPORT_FILE = os.path.join(DEPLOY_PATH, "status_report.txt")

LEARNING_INTERVAL = 0       # 实时学习模式，无间隔
STATUS_INTERVAL = 1800      # 状态汇报间隔（秒，30分钟）
EVOLVE_INTERVAL = 40000     # 每N条进化一次
EVOLVE_GENERATIONS = 20     # 每次进化代数
BATCH_SIZE = 40000          # 批量大小（40000×1024矩阵，OpenMP充分并行）
PARALLEL_WORKERS = 8        # 并行【编码+推理】子进程数
BATCH_LOG_EVERY = 1         # 每N个batch打一次日志
SAVE_EVERY = 40000          # 每N条保存权重和状态
HTTP_TIMEOUT = 5            # 网络请求超时

# 模型维度
TEXT_DIM = 1024         # 输入维度
HIDDEN_DIM = 512        # 隐藏层1
HIDDEN2_DIM = 512       # 隐藏层2（新增加的非线性层）
OUTPUT_DIM = 1024       # 输出维度 = 全量编码重建
HEAD_ACTIONS = 1024     # 决策头维度 = 全量编码
MEM_DIM = 256           # C++ retrieved_mem 维度
MEM_LOSS_WEIGHT = 0.3   # retrieved_mem 损失权重
CONTRASTIVE_WEIGHT = 1.2    # 对比损失权重（从0.8提升至1.2，增强域间推散力）
VICREG_VAR_WEIGHT = 2.5     # VICReg方差正则化权重（5x临时提权，加速方差恢复）
VICREG_GAMMA = 1.0          # VICReg目标标准差阈值（γ，新增）
# ── 记忆能量泵（Memory Energy Pump）──
MEM_NORM_PUMP_WEIGHT = 0.5   # M_V范数泵权重：拉||M_V||向1.0
MEM_NORM_TARGET = 1.0        # M_V目标平均行范数
MEM_DIVERSITY_WEIGHT = 0.3   # M_V多样性损失权重：推散各槽向量
VOCAB_SIZE = 500        # 词汇表大小（top 500 高频字符，提高正类密度）
VOCAB_LOSS_WEIGHT = 0.5     # 词汇预测损失权重（提升至0.5，强梯度恢复词表头）
VOCAB_POS_WEIGHT = 1.0      # 正类加权（未使用）
WEIGHT_DECAY = 0.002        # L2权重衰减系数（防止权重无限增长）

# ═══ HPC专属：50%权重核爆破（物理粉碎坍塌吸引子，非Forward/梯度修补） ═══
HPC_AMPLIFY_ENABLED = False       # 停用核爆（等HPC权重覆盖）
HPC_FORCE_RAW_GATE = False        # 记忆通路正常

os.makedirs(KB_DIR, exist_ok=True)

# ── 门控温度锐化补丁 ──
from gate_patch_numpy import DynamicGateSharper

os.environ.setdefault("OMP_NUM_THREADS", "40")

RECOVERY_MODE = True          # 恢复期临时禁用SAE稀疏（全维梯度通过，var恢复后改回False）

# 词表梯度热身控制器（双重锁步热身策略）
# 阶段1: 隔离期 — 仅调W_gen权重，词表梯度不进入h3共享层
# 阶段2: Sigmoid渐进释放 — 词表梯度平滑注入h3
class VocabGradientController:
    """词表梯度热身控制器 — 防止词表BCE梯度砸碎刚恢复的流形"""
    def __init__(self, target_h3_weight=0.2, target_wgen_lr_mult=1.0,
                 warm_up_steps=3000, start_var_threshold=0.1):
        self.target_h3_weight = target_h3_weight   # 最终注入h3的词表梯度权重
        self.target_wgen_lr_mult = target_wgen_lr_mult  # W_gen的学习率乘数
        self.warm_up_steps = warm_up_steps
        self.start_var_threshold = start_var_threshold
        self.warm_up_counter = 0
        self.is_warming_up = False
        self.active = False
        self.h3_weight = 0.0     # 当前h3梯度混合权重（phase 2）
        self.wgen_lr_mult = 0.0  # 当前W_gen学习率乘数（phase 1）
    
    def update(self, current_var):
        """返回 (h3_word_weight, wgen_lr_mult)"""
        if not self.active and current_var >= self.start_var_threshold:
            self.active = True
            self.is_warming_up = True
            self.warm_up_counter = 0
            print(f"🔥 [Vocab] var={current_var:.4f} 达标，启动词表梯度热身(3000步Sigmoid)")
        
        if not self.active:
            self.h3_weight = 0.0
            self.wgen_lr_mult = 0.0
            return 0.0, 0.0
        
        if self.is_warming_up:
            self.warm_up_counter += 1
            x = 12.0 * (self.warm_up_counter / self.warm_up_steps) - 6.0
            sigmoid_factor = 1.0 / (1.0 + np.exp(-x))
            
            # 阶段1(wgen_lr_mult先行): W_gen学习率先恢复
            # 阶段2(h3_weight跟进): 词表梯度再注入共享层
            self.wgen_lr_mult = sigmoid_factor * self.target_wgen_lr_mult
            self.h3_weight = max(0.0, sigmoid_factor - 0.5) * 2.0 * self.target_h3_weight
            # h3_weight在sigmoid_factor<0.5时为0 → 自然隔离
            
            if self.warm_up_counter >= self.warm_up_steps:
                self.is_warming_up = False
                self.h3_weight = self.target_h3_weight
                self.wgen_lr_mult = self.target_wgen_lr_mult
                print(f"✅ [Vocab] 热身完成! h3_w={self.h3_weight:.3f} wgen_lr={self.wgen_lr_mult:.3f}")
            
            return self.h3_weight, self.wgen_lr_mult
        
        return self.h3_weight, self.wgen_lr_mult
    
    def status_str(self):
        if not self.active:
            return "未激活"
        if self.is_warming_up:
            pct = int(100 * self.warm_up_counter / self.warm_up_steps)
            return f"热身中({pct}%) h3_w={self.h3_weight:.3f} wgen={self.wgen_lr_mult:.3f}"
        return f"已就绪 h3_w={self.h3_weight:.3f}"

# ═══════════════════════════════════════════════════════════════
# 内置本地知识库 — 200+ 跨领域知识点
# ═══════════════════════════════════════════════════════════════
BUILTIN_KNOWLEDGE = [
    # ── 科学 ──
    "The scientific method involves observation hypothesis testing and peer review",
    "Atoms consist of protons neutrons and electrons orbiting the nucleus",
    "DNA is a double helix structure containing genetic information for all life",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "The theory of relativity states that space and time are interwoven into spacetime",
    "Quantum mechanics describes the behavior of particles at atomic scales",
    "Evolution by natural selection explains how species adapt over generations",
    "The periodic table organizes elements by atomic number and chemical properties",
    "Newton laws of motion describe the relationship between force mass and acceleration",
    "Entropy measures the disorder in a system and always increases in isolated systems",
    "The speed of light in vacuum is approximately three hundred thousand km/s",
    "Electromagnetic waves include radio microwaves infrared visible ultraviolet X-rays",
    "Plate tectonics explains the movement of Earth crust and formation of mountains",
    "The water cycle involves evaporation condensation precipitation and collection",
    "Mitochondria are the powerhouses of the cell producing ATP through respiration",
    "The human brain contains approximately eighty six billion neurons",
    "Nuclear fusion powers the sun by combining hydrogen atoms into helium",
    "The greenhouse effect traps heat in Earth atmosphere through CO2 and methane",
    "Biodiversity refers to the variety of life forms in an ecosystem",
    "Antibiotics target bacterial cell walls while viruses require different treatments",
    # ── 数学 ──
    "The Pythagorean theorem: square of hypotenuse equals sum of squares of legs",
    "Calculus was developed independently by Newton and Leibniz in the 17th century",
    "Prime numbers are integers divisible only by themselves and one",
    "The Fibonacci sequence appears in nature from sunflower seeds to galaxy spirals",
    "Complex numbers consist of real and imaginary parts represented as a plus bi",
    "Probability theory quantifies uncertainty and forms the foundation of statistics",
    "Linear algebra deals with vectors matrices and systems of linear equations",
    "Topology studies properties of space preserved under continuous deformations",
    "Game theory analyzes strategic interactions between rational decision makers",
    "Fractals are geometric patterns that repeat at every scale showing self similarity",
    "The number pi represents the ratio of a circle circumference to its diameter",
    "Binary number system uses only zero and one forming the basis of computing",
    "Differential equations model rates of change in physics engineering economics",
    "Set theory provides the foundation for all of modern mathematics",
    "Graph theory studies networks of vertices connected by edges",
    # ── 计算机科学 ──
    "Algorithms are step by step procedures for solving computational problems",
    "Artificial neural networks are inspired by the structure of biological brains",
    "The internet is a global network connecting billions of devices through TCP IP",
    "Cryptography protects information through encryption and decryption techniques",
    "Database systems organize and retrieve structured data efficiently using SQL",
    "Operating systems manage hardware resources and provide services to applications",
    "Machine learning algorithms improve performance through experience and data",
    "Object oriented programming organizes code into classes with properties and methods",
    "Cloud computing provides on demand access to computing resources over the internet",
    "Blockchain technology enables decentralized and immutable record keeping",
    "The Turing test evaluates whether a machine can exhibit intelligent behavior",
    "Big data refers to extremely large datasets requiring specialized processing tools",
    "Cybersecurity protects computer systems from theft damage and unauthorized access",
    "Open source software allows users to view modify and distribute the source code",
    "Virtual reality creates immersive computer generated environments for users",
    "Quantum computers use qubits that can exist in multiple states simultaneously",
    "Natural language processing enables computers to understand human language",
    "Edge computing processes data near the source rather than in centralized centers",
    "The semantic web aims to make internet data machine readable",
    "Reinforcement learning agents learn by interacting with an environment",
    # ── 哲学 ──
    "Epistemology is the branch of philosophy concerned with the nature of knowledge",
    "Ethics examines moral principles that govern behavior and decision making",
    "Existentialism emphasizes individual freedom choice and the search for meaning",
    "The mind body problem questions the relationship between consciousness and brain",
    "Stoicism teaches that virtue is the highest good and we accept what we cannot control",
    "Utilitarianism holds that the best action maximizes overall happiness and well being",
    "Determinism suggests that all events are caused by preceding events and natural laws",
    "Free will is the capacity to make choices not determined by prior causes",
    "Plato theory of forms proposes abstract ideas exist in a perfect realm",
    "Descartes I think therefore I am establishes certainty of self existence",
    "Kant categorical imperative: act only according to that maxim which you can will to become universal law",
    "Nietzsche concept of eternal recurrence challenges how we value our lives",
    # ── 文学与艺术 ──
    "Shakespeare wrote tragedies comedies and histories exploring human nature",
    "The Renaissance was a period of cultural rebirth in Europe from 14th to 17th century",
    "Poetry uses rhythm imagery and figurative language to evoke emotions and ideas",
    "Impressionism was an art movement focused on capturing light and momentary impressions",
    "Music theory includes concepts of harmony melody rhythm and dynamics",
    "Cinema combines visual storytelling with sound to create immersive narrative experiences",
    "Architecture reflects cultural values through the design of buildings and spaces",
    "Photography captures moments in time through manipulation of light and composition",
    "The novel as a literary form emerged in the 18th century",
    "Modern art challenged traditional notions of representation and beauty",
    # ── 历史 ──
    "The Agricultural Revolution transformed human societies from hunting to farming",
    "Ancient Egypt built pyramids as tombs for pharaohs along the Nile River",
    "The Roman Empire dominated the Mediterranean world for over five hundred years",
    "The Silk Road connected East and West enabling trade of goods ideas and culture",
    "The Industrial Revolution began in Britain and mechanized production processes",
    "World War One was triggered by the assassination of Archduke Franz Ferdinand",
    "The printing press invented by Gutenberg revolutionized the spread of information",
    "The French Revolution overthrew the monarchy and established principles of liberty",
    "The Cold War was a period of geopolitical tension between the US and Soviet Union",
    "Ancient Greece laid foundations for Western philosophy democracy and science",
    # ── 经济学 ──
    "Supply and demand determine market prices in a competitive economy",
    "Gross domestic product measures the total value of goods and services produced",
    "Inflation reduces the purchasing power of money over time",
    "Comparative advantage explains why countries benefit from international trade",
    "Monetary policy controls money supply and interest rates to manage economic growth",
    "The stock market allows companies to raise capital by selling shares to investors",
    "Behavioral economics studies how psychological factors influence economic decisions",
    "Scarcity is the fundamental economic problem of limited resources and unlimited wants",
    "Entrepreneurship drives innovation and creates new markets and industries",
    "GDP per capita is a common measure of a country standard of living",
    # ── 心理学 ──
    "Classical conditioning associates a neutral stimulus with a reflexive response",
    "Operant conditioning uses rewards and punishments to shape behavior",
    "Cognitive dissonance occurs when beliefs conflict with actions causing discomfort",
    "Maslow hierarchy of needs ranges from physiological requirements to self actualization",
    "Memory involves encoding storage and retrieval of information in the brain",
    "Personality traits can be described using the Big Five model OCEAN",
    "Confirmation bias leads people to favor information confirming existing beliefs",
    "Attachment theory describes how early relationships shape emotional development",
    "The placebo effect demonstrates the power of belief in medical treatment outcomes",
    "Emotional intelligence involves recognizing and managing emotions in self and others",
    # ── 生物学 ──
    "Cells are the basic structural and functional units of all living organisms",
    "Genes are segments of DNA that code for proteins and determine hereditary traits",
    "The immune system protects the body against pathogens through specialized cells",
    "Homeostasis maintains stable internal conditions despite external changes",
    "Enzymes are biological catalysts that speed up chemical reactions in organisms",
    "The nervous system transmits electrical signals between different body parts",
    "Stem cells can differentiate into various specialized cell types",
    "Bacteria are single celled organisms playing crucial roles in ecosystems",
    "Hormones are chemical messengers that regulate physiological processes",
    "Ecosystems consist of communities of organisms interacting with their environment",
    # ── 语言学 ──
    "Saussure distinguished between langue language system and parole speech acts",
    "Chomsky universal grammar proposes innate linguistic capacity in humans",
    "Phonetics studies speech sounds while phonology studies sound patterns in languages",
    "Syntax governs sentence structure while semantics deals with meaning",
    "Pragmatics studies how context contributes to meaning in communication",
    "Morphology analyzes the internal structure of words and meaningful units",
    # ── 神经科学 ──
    "Synaptic plasticity strengthens or weakens synapses based on activity patterns",
    "The prefrontal cortex is crucial for executive function and decision making",
    "Dopamine plays a key role in reward learning motivation and motor control",
    "Neuroplasticity allows the brain to reorganize throughout life",
    "The hippocampus is essential for forming new episodic memories",
    "fMRI measures brain activity by detecting changes in blood oxygenation",
    "Long term potentiation LTP is a cellular mechanism underlying learning and memory",
    "The default mode network is active during mind wandering and self referential thought",
    # ── 工程学 ──
    "The Carnot cycle defines the theoretical maximum efficiency of a heat engine",
    "Bernoulli principle relates fluid speed to pressure in a flowing fluid",
    "Ohm law states that current through a conductor is proportional to voltage",
    "The first law of thermodynamics is conservation of energy",
    "Feedback control systems use output measurements to adjust input signals",
    "Semiconductor doping creates n type and p type materials for electronics",
]


# ═══════════════════════════════════════════════════════════════
# 编码器
# ═══════════════════════════════════════════════════════════════
def encode_text(text: str, dim: int = 512) -> np.ndarray:
    """将文本编码为特征向量（hash-based + sinusoid）"""
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)

    n_words = min(len(words), 500)
    for i, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for j in range(8):
            idx = (h + j * 2654435761) % dim
            vec[int(idx)] += 0.03 / max(n_words / 30, 1)

    vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
    return vec / (np.linalg.norm(vec) + 1e-8)


# ═══════════════════════════════════════════════════════════════
# 多进程【编码+推理】工作器 — 子进程自己做编码和推理，0串行瓶颈
# ═══════════════════════════════════════════════════════════════
def _worker_encode_reason(args):
    """
    子进程：编码一批文本 + C++ retrieved_mem 辅助目标
    
    返回:
      X_chunk:    [N, TEXT_DIM] 文本编码
      recon_targets: [N, TEXT_DIM] 全量编码重建目标 (= X_chunk)
      mem_targets: [N, MEM_DIM] C++ 模型 retrieved_mem
      rewards:    [N] 奖励（固定0.5）
      start_idx:  int
    """
    texts_chunk, start_idx, text_dim = args
    sys.path.insert(0, DEPLOY_PATH)
    from neuroflow._core import create_multimodal
    
    # 每个worker加载一次C++模型（为retrieved_mem目标）
    cpp_model = create_multimodal(text_dim=text_dim, image_size=224, output_dim=10, quantize=True)
    
    n = len(texts_chunk)
    X_chunk = np.zeros((n, text_dim), dtype=np.float32)
    recon_targets = np.zeros((n, text_dim), dtype=np.float32)  # 全量编码重建
    mem_targets = np.zeros((n, MEM_DIM), dtype=np.float32)     # retrieved_mem

    for i in range(n):
        text = texts_chunk[i]

        # 编码
        words = text.lower().split()
        n_words = min(len(words), 500)
        scale = 0.03 / max(n_words / 30, 1)
        for j, word in enumerate(words[:n_words]):
            h = abs(hash(word)) % (2**31)
            for k in range(8):
                idx = (h + k * 2654435761) % text_dim
                X_chunk[i, int(idx)] += scale
        X_chunk[i] += np.sin(np.linspace(0, np.pi * n_words / 15, text_dim)).astype(np.float32) * 0.08
        norm = np.linalg.norm(X_chunk[i])
        if norm > 1e-8:
            X_chunk[i] /= norm
        
        # 重建目标 = 编码本身（自监督）
        recon_targets[i] = X_chunk[i].copy()
        
        # C++ 模型 forward → retrieved_mem
        try:
            out = cpp_model.forward_text(X_chunk[i:i+1])
            if hasattr(out, 'retrieved_mem') and out.retrieved_mem.size >= MEM_DIM:
                mem_targets[i] = out.retrieved_mem.flatten()[:MEM_DIM]
        except Exception:
            mem_targets[i] = 0.0

    rewards = np.full(n, 0.5, dtype=np.float32)
    return X_chunk, recon_targets, mem_targets, rewards, start_idx

# ═══════════════════════════════════════════════════════════════
# 混合数据源
# ═══════════════════════════════════════════════════════════════

class HybridDataSource:
    """混合数据源：本地知识库优先，网络 API 可选增强（带超时+重试）"""

    def __init__(self):
        self.builtin = list(BUILTIN_KNOWLEDGE)
        self.builtin_idx = 0
        self.kb_contents = []       # 全量知识内存 [content, ...]
        self.kb_content_idx = 0
        self.epoch = 0              # 当前遍历轮次
        self._start_preload()
        self._net_fail_count = 0
        self._net_backoff_until = 0

    def _start_preload(self):
        """启动时全量预读知识文件到内存（127K文件~250MB，29GB空闲无压力）"""
        t0 = time.time()
        kb_dir = KB_DIR
        if not os.path.isdir(kb_dir):
            return
        files = sorted(
            [f for f in os.listdir(kb_dir) if f.endswith('.txt')],
            reverse=True
        )
        n_total = len(files)
        print(f"  📖 加载 {n_total} 个知识文件中...", end="", flush=True)
        contents = []
        report_interval = max(n_total // 10, 1)
        for idx, fname in enumerate(files):
            path = os.path.join(kb_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read(4000).strip()  # 读满4000字符获取更多内容
                if len(text) > 20:
                    contents.append(text)
            except Exception:
                pass
            if (idx + 1) % report_interval == 0:
                pct = (idx + 1) / n_total * 100
                print(f" {pct:.0f}%", end="", flush=True)
        self.kb_contents = contents
        t = time.time() - t0
        print(f" ✅ {len(contents)} 条内容加载完成 ({t:.0f}s)", flush=True)

    def _retry_fetch(self, fetch_fn, max_retries=2, base_delay=1.0):
        """带指数退避的重试包装器"""
        for attempt in range(max_retries + 1):
            try:
                result = fetch_fn()
                if result and len(result) > 30:
                    return result
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
        return None

    def _read_kb_file(self) -> str | None:
        """从内存返回一条知识，轮次结束自动shuffle"""
        if not self.kb_contents:
            return None
        if self.kb_content_idx >= len(self.kb_contents):
            self.kb_content_idx = 0
            self.epoch += 1
            random.shuffle(self.kb_contents)  # 每轮shuffle
        text = self.kb_contents[self.kb_content_idx]
        self.kb_content_idx += 1
        return text

    def _fetch_online(self) -> str | None:
        """尝试从网络获取（带超时保护 + 指数退避）"""
        # 退避期跳过
        if time.time() < self._net_backoff_until:
            return None

        apis = [
            ("HN", lambda: self._fetch_hn()),
            ("NPR", lambda: self._fetch_npr()),
            ("Wiki", lambda: self._fetch_wiki_random()),
        ]
        random.shuffle(apis)

        for name, api in apis:
            result = self._retry_fetch(api, max_retries=2, base_delay=1.0)
            if result and len(result) > 30:
                self._net_fail_count = 0
                return f"[net] {result[:2000]}"

        # 全部失败 → 增加退避
        self._net_fail_count += 1
        if self._net_fail_count >= 10:
            backoff_sec = min(300, 10 * (2 ** min(self._net_fail_count - 10, 4)))
            self._net_backoff_until = time.time() + backoff_sec
            print(f"  ⚠️ 网络连续失败{self._net_fail_count}次，退避{backoff_sec}s", flush=True)
        return None

    def _fetch_hn(self) -> str | None:
        try:
            req = urllib.request.Request(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                headers={"User-Agent": "NeuroFlow/3.2"}
            )
            ids = json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read())
            for sid in ids[:5]:
                try:
                    req2 = urllib.request.Request(
                        f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                        headers={"User-Agent": "NeuroFlow/3.2"}
                    )
                    item = json.loads(urllib.request.urlopen(req2, timeout=HTTP_TIMEOUT).read())
                    title = item.get("title", "")
                    if title and len(title) > 10:
                        return title
                except:
                    continue
        except:
            pass
        return None

    def _fetch_npr(self) -> str | None:
        try:
            import xml.etree.ElementTree as ET
            req = urllib.request.Request(
                "https://feeds.npr.org/1001/rss.xml",
                headers={"User-Agent": "NeuroFlow/3.2"}
            )
            data = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read()
            root = ET.fromstring(data)
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                if title is not None and title.text:
                    return title.text.strip()
        except:
            pass
        return None

    def _fetch_wiki_random(self) -> str | None:
        try:
            req = urllib.request.Request(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                headers={"User-Agent": "NeuroFlow/3.2"}
            )
            data = json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read())
            extract = data.get("extract", "")
            if len(extract) > 100:
                return extract
        except:
            pass
        return None

    def get_next(self) -> tuple[str, str]:
        """
        获取下一条知识（本地高速模式，禁用网络避免GFW超时卡顿）
        """
        # 50% 知识库文件，50% 内置
        if random.random() < 0.5:
            text = self._read_kb_file()
            if text:
                return ("kb_file", text)
        # 内置知识
        if self.builtin_idx >= len(self.builtin):
            self.builtin_idx = 0
            random.shuffle(self.builtin)
        text = self.builtin[self.builtin_idx]
        self.builtin_idx += 1
        return ("builtin", text)

    def get_batch(self, n: int = 100) -> list[tuple[str, str]]:
        """批量获取N条知识（预读缓冲，减少逐个I/O开销）"""
        batch = []
        for _ in range(n):
            batch.append(self.get_next())
        return batch

    def size(self) -> dict:
        return {
            "builtin": len(self.builtin),
            "kb_files": len(self.kb_contents),
        }


# ═══════════════════════════════════════════════════════════════
# 守护进程
# ═══════════════════════════════════════════════════════════════
class NeuroFlowDaemonV3:
    """NeuroFlow v3 自主进化守护进程"""

    def __init__(self):
        self._init_model()
        self.data_source = HybridDataSource()

        self._last_slot_prune_batch = 0  # 记录上次修剪的批次
        
        # 状态
        self.state = {
            "started": datetime.now().isoformat(),
            "topics": 0,
            "source_stats": {"kb_file": 0, "builtin": 0, "net": 0},
            "evolutions": 0,
            "fitness": 0.0,
            "knowledge_files": 0,
            "errors": 0,
            "last_activity": "",
            "total_loss": 0.0,
            "train_steps": 0,
        }
        self._load_state()

        # 统计
        self.cycle_count = 0
        self.last_status = time.time()
        
        # 词表梯度热身控制器（var达标后渐进引入词表损失梯度）
        self.vocab_controller = VocabGradientController(
            target_h3_weight=0.2,
            warm_up_steps=3000,
            start_var_threshold=0.1
        )

    def _init_model(self):
        """初始化模型 + W_h2加深 + W_gen词汇预测 + vocab加载"""
        from neuroflow._core import create_multimodal
        from neuroflow.trainable_head import TrainableHead
        
        # NeuroSymbolicReasoner 兼容包装（新cognition.py使用ReasoningLoop）
        from neuroflow.cognition import ReasoningLoop
        class _NSR(ReasoningLoop):
            def reason(self, x, max_steps=5, verbose=False):
                trace = super().reason(x, max_steps=max_steps, verbose=verbose)
                trace.final_confidence = trace.thoughts[-1].confidence if trace.thoughts else 0.5
                trace.final_action = trace.final_decision
                trace.steps = trace.thoughts
                trace.elapsed_ms = trace.convergence_time
                trace.explanation = ""
                return trace
        
        self.model = create_multimodal(
            text_dim=TEXT_DIM, image_size=224, output_dim=OUTPUT_DIM, quantize=True
        )
        self.reasoner = _NSR(self.model)
        self.head = TrainableHead(
            self.model, hidden_dim=HIDDEN_DIM, n_actions=HEAD_ACTIONS, lr=0.01,
            input_dim=TEXT_DIM
        )
        
        # 初始化方法: 跳过所有权重初始化，由 __init__ 中的 _init_model 完成
        pass
        
        # ── 可学习输入投影 W_embed（注意力替换：hash→可学习特征重排）──
        scale_emb = np.sqrt(2.0 / TEXT_DIM)
        self.W_embed = np.random.randn(TEXT_DIM, TEXT_DIM).astype(np.float32) * 0.01  # 小初始化，从零学起

        # ── Gated Memory Bank（替代 W_h + W_h2 的隐层加深）──
        # 24个记忆槽，每个256维，减少槽数集中训练强度
        MEM_SLOTS = 32          # 记忆槽数（从24升至32，提升容量）
        MEM_DIM_IN = 256        # 键/值维度
        # 记忆键（L2范数=1，投影到单位球面）
        K_init = np.random.randn(MEM_SLOTS, MEM_DIM_IN).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init
        # 记忆值（初始化为小随机，从头学习）
        self.M_V = np.random.randn(MEM_SLOTS, MEM_DIM_IN).astype(np.float32) * 0.01
        # 查询投影: h1(512) → Q(MEM_DIM_IN=256)
        scale_q = np.sqrt(2.0 / HIDDEN_DIM)
        self.W_q = np.random.randn(HIDDEN_DIM, MEM_DIM_IN).astype(np.float32) * scale_q
        # 门控: h1(512) → gate(512)
        self.W_gate = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
        self.b_gate = np.random.randn(1, HIDDEN_DIM).astype(np.float32) * 0.01
        # ── 门控温度锐化（余弦退火） ──
        self.gate_sharper = DynamicGateSharper(start_tau=0.2, target_tau=1.0, duration_topics=500000)
        self.gate_tau_active = False  # 初始关闭，等M_V冲上0.5后激活
        # 记忆输出投影: MEM_DIM_IN(256) → HIDDEN(512)
        scale_out = np.sqrt(2.0 / MEM_DIM_IN)
        self.W_mem_out = np.random.randn(MEM_DIM_IN, HIDDEN_DIM).astype(np.float32) * scale_out
        
        # ── 新增：retrieved_mem 头 W_m ──
        scale_m = np.sqrt(2.0 / HIDDEN2_DIM)
        self.W_m = np.random.randn(HIDDEN2_DIM, MEM_DIM).astype(np.float32) * scale_m
        self.b_m = np.zeros((1, MEM_DIM), dtype=np.float32)
        
        # ── 词汇预测头 W_gen（HPC除颤：正交初始化粉碎对称性）──
        Q, _ = np.linalg.qr(np.random.randn(HIDDEN2_DIM, HIDDEN2_DIM).astype(np.float32))
        self.W_gen = Q[:, :VOCAB_SIZE].astype(np.float32) * 1.5  # (512, 500) 正交列×1.5
        self.b_gen = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        
        # ── 独立词表头（不共享隐层梯度） ──
        scale_vi = np.sqrt(2.0 / HIDDEN2_DIM)
        self.V_in = np.random.randn(HIDDEN2_DIM, 256).astype(np.float32) * scale_vi
        self.V_out = np.random.randn(256, VOCAB_SIZE).astype(np.float32) * scale_vi
        self.V_bias = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        self._vocab_ready = False  # 训练完成标记
        
        # ── 加载字符级词汇表 ──
        self.char_vocab = []
        if os.path.exists(VOCAB_FILE):
            import json
            with open(VOCAB_FILE, encoding='utf-8') as f:
                self.char_vocab = json.load(f)
            print(f"  📖 字符词表: {len(self.char_vocab)} 个, 从 {VOCAB_FILE}")
        else:
            # 降级: 使用ASCII字符+基本汉字
            self.char_vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}")
            print(f"  ⚠️  未找到字符词表, 使用基本ASCII字符表({len(self.char_vocab)}个)")
        
        # 预分配批量矩阵
        self._X_batch = np.zeros((BATCH_SIZE, TEXT_DIM), dtype=np.float32)
        self._Y_batch = np.zeros((BATCH_SIZE, TEXT_DIM), dtype=np.float32)   # 编码重建目标
        self._M_batch = np.zeros((BATCH_SIZE, MEM_DIM), dtype=np.float32)    # retrieved_mem
        self._W_batch = np.zeros((BATCH_SIZE, VOCAB_SIZE), dtype=np.float32) # 词汇标记
        self._rewards_batch = np.zeros(BATCH_SIZE, dtype=np.float32)

    def _load_state(self):
        """加载持久化状态"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    saved = json.load(f)
                for key in ["topics", "evolutions", "fitness", "knowledge_files",
                           "errors", "train_steps", "total_loss"]:
                    if key in saved:
                        self.state[key] = saved[key]
                if "source_stats" in saved:
                    self.state["source_stats"] = saved["source_stats"]
            except Exception:
                pass

        # 加载训练权重（带形状校验，自动跳过不匹配的参数）
        if os.path.exists(WEIGHTS_FILE):
            try:
                data = np.load(WEIGHTS_FILE)
                for attr, saved_key in [
                    ("head.W_d", "W_d"), ("head.b_d", "b_d"),
                    ("head.W_v", "W_v"), ("head.b_v", "b_v"),
                    ("head.W_p", "W_p"),
                    ("M_K", "M_K"), ("M_V", "M_V"),
                    ("W_q", "W_q"), ("W_gate", "W_gate"),
                    ("b_gate", "b_gate"), ("W_mem_out", "W_mem_out"),
                    ("W_m", "W_m"), ("b_m", "b_m"),
                    ("W_gen", "W_gen"), ("b_gen", "b_gen"),
                    ("W_embed", "W_embed"),
                    ("V_in", "V_in"), ("V_out", "V_out"), ("V_bias", "V_bias"),
                ]:
                    if saved_key in data:
                        saved_arr = data[saved_key]
                        curr_arr = getattr(self, attr.split(".")[0]) if "." not in attr else getattr(getattr(self, attr.split(".")[0]), attr.split(".")[1])
                        if saved_arr.shape == curr_arr.shape:
                            if "." not in attr:
                                setattr(self, attr, saved_arr)
                            else:
                                parent, child = attr.split(".")
                                setattr(getattr(self, parent), child, saved_arr)
            except Exception:
                pass
        
        # ═══ 终极方案：50%权重核爆破 — 物理粉碎坍塌吸引子 ═══
        if HPC_AMPLIFY_ENABLED:
            print("🚨 [NUCLEAR] 启动50%权重级核爆破！物理粉碎坍塌吸引子...")
            target_names = ['W_embed', 'W_p', 'W_gate', 'W_mem_out', 'M_K', 'M_V']
            for name in target_names:
                if hasattr(self, name):
                    param = getattr(self, name)
                    param_std = max(float(np.std(param)), 1e-5)
                    noise = np.random.randn(*param.shape).astype(np.float32) * (param_std * 0.50)
                    param += noise
                    print(f"       💥 已向层 [{name}] 注入 50% 噪声 (std={param_std*0.50:.5f})")
                elif hasattr(self.head, name):
                    param = getattr(self.head, name)
                    param_std = max(float(np.std(param)), 1e-5)
                    noise = np.random.randn(*param.shape).astype(np.float32) * (param_std * 0.50)
                    param += noise
                    print(f"       💥 已向层 [head.{name}] 注入 50% 噪声 (std={param_std*0.50:.5f})")
                else:
                    print(f"       ⚠ 未找到层 [{name}] 或 [head.{name}]，跳过")
            # W_p完全替换为放大随机矩阵（scale×5，提升h3方差至0.01+）
            W_p_fresh = np.random.randn(TEXT_DIM, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0/TEXT_DIM) * 5.0
            self.head.W_p = W_p_fresh
            print(f"       🔄 层 [head.W_p] 完全替换为放大5x随机矩阵 (scale={np.sqrt(2.0/TEXT_DIM)*5:.5f})")
            # W_gen: 强正交化(gain=1.5) + bias清零
            Q, _ = np.linalg.qr(np.random.randn(HIDDEN2_DIM, HIDDEN2_DIM).astype(np.float32))
            self.W_gen = Q[:, :VOCAB_SIZE].astype(np.float32) * 1.5
            self.b_gen = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
            print(f"       🔮 W_gen 正交重置 gain=1.5, bias清零")
            print("✅ [NUCLEAR] 拓扑流形物理粉碎完毕。")
    
    def _save_state(self):
        """保存状态"""
        self.state["source_stats"] = self.state.get("source_stats", {})
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _save_weights(self):
        """保存训练权重（含 W_embed,词表头等全部参数）"""
        w = {
            "W_d": self.head.W_d.copy(),
            "b_d": self.head.b_d.copy(),
            "W_v": self.head.W_v.copy(),
            "b_v": self.head.b_v.copy(),
            "W_p": self.head.W_p.copy(),
            "M_K": self.M_K.copy(),
            "M_V": self.M_V.copy(),
            "W_q": self.W_q.copy(),
            "W_gate": self.W_gate.copy(),
            "b_gate": self.b_gate.copy(),
            "W_mem_out": self.W_mem_out.copy(),
            "W_m": self.W_m.copy(),
            "b_m": self.b_m.copy(),
            "W_gen": self.W_gen.copy(),
            "b_gen": self.b_gen.copy(),
            "W_embed": self.W_embed.copy(),
            "V_in": self.V_in.copy(),
            "V_out": self.V_out.copy(),
            "V_bias": self.V_bias.copy(),
        }
        np.savez(WEIGHTS_FILE, **w)

    def _learn_one(self) -> dict:
        """学习一个知识点"""
        source, text = self.data_source.get_next()

        # 编码 → 推理
        x = encode_text(text, TEXT_DIM).reshape(1, -1)
        trace = self.reasoner.reason(x, max_steps=5)

        # 评估
        conf = trace.final_confidence if trace.steps else 0.5
        reward = conf

        # SGD 训练（自蒸馏）
        if trace.final_action is not None and trace.total_steps > 0:
            target = int(np.argmax(trace.final_action))
            result = self.head.train_step(x, target, reward)
            self.state["train_steps"] += 1
            self.state["total_loss"] += result["loss"]
        else:
            result = {"loss": 0.0, "predicted_action": -1}

        # 统计
        source_stats = self.state.setdefault("source_stats", {})
        source_stats[source] = source_stats.get(source, 0) + 1

        return {
            "source": source,
            "text": text[:80],
            "conf": conf,
            "reward": reward,
            "loss": result.get("loss", 0.0),
        }

    def _train_vocab_separately(self):
        """独立训练词表头（不反向传播到共享层）"""
        if not self.char_vocab or len(self.char_vocab) < 10:
            return {"vocab_acc": 0.0, "vocab_loss": 0.0}
        
        # 收集一批样本的 h3_relu 特征 + 字符目标
        N = min(20000, len(self.data_source.kb_contents))
        items = self.data_source.get_batch(N)
        
        # 编码（使用已有编码函数）
        X = np.zeros((N, TEXT_DIM), dtype=np.float32)
        for i in range(N):
            text = items[i][1]
            X[i] = encode_text(text, TEXT_DIM)
        
        # 前向（通过Gated Memory Bank到h3，冻结共享层）
        h1 = X @ self.head.W_p
        h1_relu = np.maximum(h1, 0)
        Q_v = h1_relu @ self.W_q
        K_norm_v = self.M_K / (np.linalg.norm(self.M_K, axis=1, keepdims=True) + 1e-8)
        scores_v = Q_v @ K_norm_v.T
        temp_v = 8.0
        scores_max_v = np.max(scores_v, axis=1, keepdims=True)
        scores_exp_v = np.exp(temp_v * (scores_v - scores_max_v))
        topk_v = 6
        scores_topk_v = np.partition(scores_exp_v, -topk_v, axis=1)[:, -topk_v:-topk_v+1].min(axis=1, keepdims=True)
        scores_exp_v = scores_exp_v * (scores_exp_v >= scores_topk_v)
        attn_v = scores_exp_v / (np.sum(scores_exp_v, axis=1, keepdims=True) + 1e-8)
        mem_read_v = attn_v @ self.M_V
        # ═══ 二阶段词表头训练 ═══
        mem_feat_v = mem_read_v @ self.W_mem_out
        gate_logits_v = h1_relu @ self.W_gate + self.b_gate
        if self.gate_tau_active:
            gate_v = self.gate_sharper.patch_forward(gate_logits_v)
        else:
            gate_v = 1.0 / (1.0 + np.exp(-gate_logits_v))
        h_mem_v = gate_v * h1_relu + (1.0 - gate_v) * mem_feat_v
        h3 = np.maximum(h_mem_v, 0)
        
        # 构造字符目标
        targets = np.zeros((N, VOCAB_SIZE), dtype=np.float32)
        for i in range(N):
            text = items[i][1]
            chars = set(text)
            for ch in chars:
                if ch in self.char_vocab:
                    targets[i, self.char_vocab.index(ch)] = 1.0
        
        # 训练独立词表头 V_in → ReLU → V_out
        lr_v = 0.005
        epochs = 5
        final_loss = 0.0
        for ep in range(epochs):
            # 前向
            v_hidden = h3 @ self.V_in  # [N, 256]
            v_hidden_r = np.maximum(v_hidden, 0)
            v_logits = v_hidden_r @ self.V_out + self.V_bias  # [N, 500]
            v_probs = 1.0 / (1.0 + np.exp(-v_logits))
            
            # 损失
            loss = np.mean(-targets * np.log(v_probs + 1e-8) - (1 - targets) * np.log(1 - v_probs + 1e-8))
            
            # 梯度（仅更新V_in, V_out, V_bias）
            grad = (v_probs - targets) / N
            grad_hidden = grad @ self.V_out.T * (v_hidden > 0).astype(np.float32)
            
            self.V_in -= lr_v * (h3.T @ grad_hidden)
            self.V_out -= lr_v * (v_hidden_r.T @ grad)
            self.V_bias -= lr_v * np.sum(grad, axis=0, keepdims=True)
            
            final_loss = float(loss)
        
        # 计算准确率（top-5命中率）
        top5 = np.argsort(v_probs, axis=1)[:, -5:]
        hits = 0
        total = 0
        for i in range(min(N, 1000)):
            actual = set(np.where(targets[i] > 0.5)[0])
            predicted = set(top5[i])
            hits += len(actual & predicted)
            total += len(actual) if len(actual) > 0 else 1
        acc = hits / max(total, 1)
        
        self._vocab_ready = True
        return {"vocab_acc": acc, "vocab_loss": final_loss}

    def _learn_batch(self):
        """
        批量学习 — 多进程并行编码 → 全量重建 + retrieved_mem + 非线性
        
        新架构:
          X → W_p → ReLU → W_h → ReLU → h2
            ├─ W_d → 编码重建[1024]  (MSE损失A)
            ├─ W_m → retrieved_mem   (MSE损失B)
            └─ W_v → value           (MSE损失C)
        """
        items = self.data_source.get_batch(BATCH_SIZE)
        batch_size = BATCH_SIZE
        n_workers = min(PARALLEL_WORKERS, batch_size)
        source_stats = self.state.setdefault("source_stats", {})

        for source, _ in items:
            source_stats[source] = source_stats.get(source, 0) + 1

        # 分发给子进程
        chunk_size = (batch_size + n_workers - 1) // n_workers
        chunks = []
        for w in range(n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, batch_size)
            if start >= batch_size:
                break
            texts = [t for _, t in items[start:end]]
            chunks.append((texts, start, TEXT_DIM))

        Y_batch = self._Y_batch    # [BATCH, 1024] 编码重建目标
        M_batch = self._M_batch    # [BATCH, 256]  retrieved_mem
        R_batch = self._rewards_batch

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_encode_reason, c): c for c in chunks}
            for f in as_completed(futures):
                X_chunk, recon_chunk, mem_chunk, r_chunk, start = f.result()
                n = len(recon_chunk)
                self._X_batch[start:start+n] = X_chunk
                Y_batch[start:start+n] = recon_chunk
                M_batch[start:start+n] = mem_chunk
                for j in range(n):
                    R_batch[start + j] = r_chunk[j]

        X = self._X_batch[:batch_size]   # [N, 1024] 编码
        Y = Y_batch[:batch_size]         # [N, 1024] 重建目标
        M = M_batch[:batch_size]         # [N, 256]  retrieved_mem
        R = R_batch[:batch_size]         # [N]

        # ═══ 掩码 + 噪声: 随机遮住部分维度 + 高斯噪声 ═══
        mask = np.random.rand(batch_size, TEXT_DIM) > MASK_RATIO  # (1-MASK_RATIO)%可见, MASK_RATIO%遮住
        X_masked = X * mask.astype(np.float32)
        X_masked += np.random.randn(batch_size, TEXT_DIM).astype(np.float32) * INPUT_NOISE
        
        # ═══ 前向传播 ═══
        # 输入投影: X_masked → W_embed → ReLU → 残差连接
        X_proj = X_masked @ self.W_embed  # 可学习特征重排
        X_proj = np.maximum(X_proj, 0)
        X_in = X_masked + X_proj * 0.1    # 残差（防破坏原始hash信息）
        
        # 第一层: X_in → W_p → h1 → ReLU
        h1 = X_in @ self.head.W_p                   # [N, 512]
        h1_relu = np.maximum(h1, 0)                      # ReLU
        
        # 第二层: Gated Memory Bank（替代 W_h → W_h2 加深）
        # 原理: 64个记忆槽 × 256维，可学习键值对
        # 输入查询记忆库 → 注意力读出 → 门控融合
        Q = h1_relu @ self.W_q                           # [N, 256] 查询向量
        K_norm = self.M_K / (np.linalg.norm(self.M_K, axis=1, keepdims=True) + 1e-8)  # [64, 256]
        scores = Q @ K_norm.T                            # [N, 64] 与64个记忆槽匹配
        # 只保留top-6的注意力（针对24槽）
        temp = 8.0
        scores_max = np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(temp * (scores - scores_max))
        # 只保留top-8的注意力
        topk = 6
        scores_topk = np.partition(scores_exp, -topk, axis=1)[:, -topk:-topk+1].min(axis=1, keepdims=True)
        scores_exp = scores_exp * (scores_exp >= scores_topk)
        attn = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-8)  # [N, 64]
        
        # ── 记忆槽使用跟踪（用于闲置槽自动替换） ──
        slot_hits = (attn > 0.01).sum(axis=0)  # 每个槽被多少样本激活
        if not hasattr(self, '_slot_hits'):
            self._slot_hits = np.zeros(attn.shape[1], dtype=np.int32)
        self._slot_hits += np.clip(slot_hits.astype(np.int32), 0, 1)  # 槽在批次中被用过就算1次
        
        mem_read = attn @ self.M_V                       # [N, 256] 读记忆值
        mem_feat = mem_read @ self.W_mem_out             # [N, 512] 投影到隐层
        # 门控融合: 学习每个维度该用多少原始特征 vs 记忆
        # ── 门控计算（可选温度锐化）──
        gate_logits = h1_relu @ self.W_gate + self.b_gate
        if self.gate_tau_active:
            tau = self.gate_sharper.step(self.state['train_steps'])
            gate = self.gate_sharper.patch_forward(gate_logits)
        else:
            gate = 1.0 / (1.0 + np.exp(-gate_logits))  # [N, 512]
        if HPC_FORCE_RAW_GATE:
            gate.fill(1.0)  # 旁路记忆坍塌，100%原始特征
        h_mem = gate * h1_relu + (1.0 - gate) * mem_feat   # [N, 512] 门控融合
        h3 = np.maximum(h_mem, 0)                          # [N, 512] 最终隐层
        h3_relu = h3  # 一致（已非负）
        
        # ═══ LayerNorm 前归一化（修复：低方差下SAE随机路由问题）═══
        # 将 h3 归一化为标准正态，确保SAE的top-k选择基于相对结构而非噪声幅度
        h3_mean = np.mean(h3_relu, axis=1, keepdims=True)
        h3_std = np.std(h3_relu, axis=1, keepdims=True) + 1e-5
        h3_normed = (h3_relu - h3_mean) / h3_std  # [N, 512] ~ N(0,1)
        
        # ═══ Sparse Autoencoder 自适应稀疏度 (v5.0: 输入熵越高→激活越多) ═══
        # 基础top-k: 65 (12.7% of 512)
        K_BASE = 65
        K_MIN, K_MAX = 40, 120
        # 使用归一化后的值计算熵（消除幅度影响）
        h3_softmax = np.exp(h3_normed - np.max(h3_normed, axis=-1, keepdims=True))
        h3_softmax = h3_softmax / (np.sum(h3_softmax, axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(h3_softmax * np.log(h3_softmax + 1e-8), axis=-1)  # [N]
        entropy_norm = entropy / np.log(512)  # 归一化到 [0, 1]
        # 记忆槽使用率补偿: 活跃槽多→更多SAE通道
        if hasattr(self, 'mem_usage'):
            slot_bonus = int(10 * self.mem_usage)  # 最多+10
        else:
            slot_bonus = 0
        k_per_sample = (K_MIN + (K_MAX - K_MIN) * entropy_norm + slot_bonus).astype(np.int32)
        k_per_sample = np.clip(k_per_sample, K_MIN, K_MAX)
        # 取平均值作为该batch的统一top-k（稳定训练）
        K_DYNAMIC = int(np.mean(k_per_sample))
        
        h3_normed_abs = np.abs(h3_normed)
        h3_thresh = np.partition(h3_normed_abs, -K_DYNAMIC, axis=1)[:, -K_DYNAMIC:-K_DYNAMIC+1]
        self._sae_mask = (h3_normed_abs >= h3_thresh).astype(np.float32)
        self._sae_k = K_DYNAMIC  # 记录供日志使用
        # 恢复期暂时跳过SAE mask（全512维通过，加速var恢复）
        if not RECOVERY_MODE:
            h3_relu = h3_relu * self._sae_mask
        
        # ═══ HPC专属：输入嵌入层噪声注入（打破对称坍塌，通过全路径传播） ═══
        # [已移除: 前7轮证明Forward/梯度层面修补无效，改用权重级核爆破]
        
        # 输出头
        recon = h3_relu @ self.head.W_d + self.head.b_d  # [N, 1024] 编码重建
        mem_pred = h3_relu @ self.W_m + self.b_m         # [N, 256]  retrieved_mem
        val = h3_relu @ self.head.W_v + self.head.b_v    # [N, 1]    价值
        word_logits = h3_relu @ self.W_gen + self.b_gen  # [N, 2000] 词汇预测

        # ═══ 计算字符级词汇目标 ═══
        # 从文本中提取所有出现的字符，标记哪些在词表中
        c_batch = self._W_batch[:batch_size]
        c_batch.fill(0)
        for i in range(min(len(items), batch_size)):
            text = items[i][1] if isinstance(items[i], (list, tuple)) else items[i]
            # 收集文本中所有出现在字符词表的字符
            chars_seen = set()
            for ch in text:
                if ch in self.char_vocab:
                    chars_seen.add(ch)
            for ch in chars_seen:
                idx = self.char_vocab.index(ch)
                c_batch[i, idx] = 1.0

        # 学习率退火: 每个epoch下降5%
        base_lr = 0.01
        epoch = self.data_source.epoch
        self.head.lr = max(0.0005, base_lr * (0.95 ** epoch))

        # ═══ 损失 ═══
        N = batch_size
        lr = self.head.lr
        shared_lr = lr * 0.2  # 共享层学习率 = 输出头的 1/5（防止词表头梯度冲垮共享层）
        r_arr = R.reshape(-1, 1).astype(np.float32)
        
        recon_mse = float(np.mean((recon - Y) ** 2))
        mem_mse = float(np.mean((mem_pred - M) ** 2))
        val_mse = float(np.mean((val - r_arr) ** 2))
        
        # 词汇BCE损失（正类加权用于监控，梯度用无加权版保证稳定）
        word_sigmoid = 1.0 / (1.0 + np.exp(-word_logits))  # sigmoid
        
        # —— 加权BCE损失（仅监控，不参与梯度）—— 
        pos_mask = c_batch.astype(np.float32)
        neg_mask = (1 - c_batch).astype(np.float32)
        pos_per_sample = np.sum(pos_mask, axis=1, keepdims=True).clip(1)
        neg_per_sample = np.sum(neg_mask, axis=1, keepdims=True).clip(1)
        pos_weight = VOCAB_POS_WEIGHT * (neg_per_sample / pos_per_sample)
        word_bce_weighted = float(np.mean(
            pos_weight * (-pos_mask * np.log(word_sigmoid + 1e-8))
            + (-neg_mask * np.log(1 - word_sigmoid + 1e-8))
        ))
        # 无加权BCE（用于展示和梯度）
        word_bce = float(np.mean(
            -c_batch * np.log(word_sigmoid + 1e-8) - (1 - c_batch) * np.log(1 - word_sigmoid + 1e-8)
        ))
        
        # 对比损失: 最大化隐状态方差
        h_mean = np.mean(h3_relu, axis=0, keepdims=True)
        h_centered = h3_relu - h_mean
        h_var = float(np.mean(h_centered ** 2))
        contrastive_loss = -CONTRASTIVE_WEIGHT * h_var
        
        # ═══ 记忆能量泵 Loss ═══
        # M_V范数泵：目标每行范数 = MEM_NORM_TARGET
        mv_row_norms = np.linalg.norm(self.M_V, axis=1)  # [32]
        mv_mean_norm = float(np.mean(mv_row_norms))
        mem_norm_pump = MEM_NORM_PUMP_WEIGHT * max(0, MEM_NORM_TARGET - mv_mean_norm)
        # M_V多样性：每槽差异度（推散各槽向量）
        M_V_centered = self.M_V - np.mean(self.M_V, axis=0, keepdims=True)
        mv_var = float(np.mean(M_V_centered ** 2))
        mem_diversity = -MEM_DIVERSITY_WEIGHT * mv_var
        # 总能量泵损失（仅监控，梯度手动注入M_V更新）
        mem_pump_full = mem_norm_pump + mem_diversity
        
        loss = recon_mse + MEM_LOSS_WEIGHT * mem_mse + 0.1 * val_mse + contrastive_loss + VOCAB_LOSS_WEIGHT * word_bce

        # 记录MSE供进化
        if not hasattr(self, '_recent_mse'):
            self._recent_mse = []
            self._recent_var = []
            self._recent_word_bce = []
        self._recent_mse.append(recon_mse)
        self._recent_var.append(h_var)
        self._recent_word_bce.append(word_bce)

        # 梯度计算 — word用无加权BCE梯度保证稳定
        grad_recon = 2 * (recon - Y) / N                         # [N, 1024]
        grad_mem = 2 * MEM_LOSS_WEIGHT * (mem_pred - M) / N       # [N, 256]
        grad_val = 2 * 0.1 * (val - r_arr) / N                    # [N, 1]
        
        # ═══ 词表梯度热身：双重锁步 ═══
        # h3_weight: 词表梯度注入共享层的权重（phase 2）
        # wgen_lr_mult: W_gen学习率乘数（phase 1先行）
        h3_word_weight, wgen_lr_mult = self.vocab_controller.update(h_var)
        
        # 词表梯度（始终满梯度计算，但分两路）
        grad_word_raw = (word_sigmoid - c_batch) / N  # [N, 500] 原始BCE梯度
        grad_word = h3_word_weight * grad_word_raw     # 注入h3的梯度（phase 2渐进）
        
        # 输出层更新 (使用 h3_relu, +L2 weight decay)
        self.head.W_d -= lr * ((h3_relu.T @ grad_recon) + WEIGHT_DECAY * self.head.W_d)
        self.head.b_d -= lr * np.sum(grad_recon, axis=0, keepdims=True)
        self.W_m -= lr * ((h3_relu.T @ grad_mem) + WEIGHT_DECAY * self.W_m)
        self.b_m -= lr * np.sum(grad_mem, axis=0, keepdims=True)
        self.head.W_v -= lr * (0.1 * (h3_relu.T @ grad_val) + WEIGHT_DECAY * self.head.W_v)
        self.head.b_v -= lr * 0.1 * np.sum(grad_val, axis=0, keepdims=True)
        # W_gen: 始终满梯度更新（不受h3_word_weight限制），仅受wgen_lr_mult控速
        self.W_gen -= lr * wgen_lr_mult * ((h3_relu.T @ grad_word_raw) + WEIGHT_DECAY * self.W_gen)
        self.b_gen -= lr * wgen_lr_mult * np.sum(grad_word_raw, axis=0, keepdims=True)

        # 对比损失梯度（推散隐状态）
        grad_contrastive = -2 * CONTRASTIVE_WEIGHT / N * h_centered  # [N, 512]

        # ═══ VICReg方差正则化梯度（逐维度标准差铰链损失）═══
        per_dim_std = np.sqrt(np.mean(h_centered ** 2, axis=0) + 1e-4)  # [D]
        vicreg_mask = (per_dim_std < VICREG_GAMMA).astype(np.float32)   # [D] hinge mask
        vicreg_grad = -VICREG_VAR_WEIGHT / N * h_centered / (per_dim_std + 1e-8) * vicreg_mask  # [N, D]

        # ═══ 恢复期：直接噪声注入h3提升方差（跳过所有补偿机制）═══
        if RECOVERY_MODE:
            # 高斯噪声直接注入h3，迫使模型学习区分信号vs噪声
            noise_std = 0.05 * np.clip(1.0 - h_var * 100, 0.05, 1.0)  # var越低噪声越大
            h3_relu += np.random.randn(*h3_relu.shape).astype(np.float32) * noise_std

        # Gated Memory Bank 梯度（替代 W_h2 → W_h 的反向传播）
        grad_h3 = (grad_recon @ self.head.W_d.T) + (grad_mem @ self.W_m.T) \
                 + (grad_val @ self.head.W_v.T) * 0.1 + (grad_word @ self.W_gen.T)
        grad_h3 += grad_contrastive  # 对比度梯度
        # SAE 稀疏掩码梯度（恢复期跳过）
        if hasattr(self, '_sae_mask') and not RECOVERY_MODE:
            grad_h3 = grad_h3 * self._sae_mask
            self._sae_mask = None  # 用后清理
        elif RECOVERY_MODE:
            self._sae_mask = None  # 恢复期只清理不mask
        # ═══ 梯度裁剪 ═══
        h3_norm = np.linalg.norm(grad_h3)
        if h3_norm > 1.0:
            grad_h3 *= 1.0 / h3_norm
        # VICReg梯度推散已通过直接W_p注入完成，不再添加至grad_h3
        
        # h3 = ReLU(h_mem) → grad_h_mem = grad_h3 * (h_mem > 0)
        grad_h_mem = grad_h3 * (h_mem > 0).astype(np.float32)
        
        # ── 门控路径梯度 ──
        # h_mem = gate * h1_relu + (1-gate) * mem_feat
        d_gate = grad_h_mem * (h1_relu - mem_feat)
        # gate = sigmoid(gate_logits / τ) 的链式法则
        if self.gate_tau_active:
            d_sigmoid = self.gate_sharper.patch_backward(d_gate, gate)
        else:
            d_sigmoid = d_gate * gate * (1.0 - gate)
        self.W_gate -= shared_lr * ((h1_relu.T @ d_sigmoid) + WEIGHT_DECAY * self.W_gate)
        self.b_gate -= shared_lr * np.sum(d_sigmoid, axis=0, keepdims=True)
        # h1 通过 gate 路径的梯度
        d_h1_gate = d_sigmoid @ self.W_gate.T
        # h1 通过直接路径的梯度
        d_h1_direct = grad_h_mem * gate
        
        # ── 记忆读出路径梯度 ──
        d_mem_feat = grad_h_mem * (1.0 - gate)
        self.W_mem_out -= shared_lr * ((mem_read.T @ d_mem_feat) + WEIGHT_DECAY * self.W_mem_out)
        d_mem_read = d_mem_feat @ self.W_mem_out.T
        
        # mem_read = attn @ M_V
        d_attn = d_mem_read @ self.M_V.T
        # ═══ 记忆能量泵梯度注入 M_V ═══
        # 范数泵梯度：指向放大方向（梯度下降取负→推离原点）
        mv_row_norms = np.linalg.norm(self.M_V, axis=1, keepdims=True) + 1e-8
        mv_norm_grad = -MEM_NORM_PUMP_WEIGHT * self.M_V / mv_row_norms  # 负号=推离原点
        mv_norm_grad *= (mv_row_norms < MEM_NORM_TARGET).astype(np.float32)
        # 多样性梯度：推散各槽
        M_V_centered = self.M_V - np.mean(self.M_V, axis=0, keepdims=True)
        mv_div_grad = -2 * MEM_DIVERSITY_WEIGHT / 32 * M_V_centered
        # 合并能量泵梯度
        mv_pump_grad = mv_norm_grad + mv_div_grad
        self.M_V -= shared_lr * ((attn.T @ d_mem_read) + WEIGHT_DECAY * self.M_V + mv_pump_grad)
        
        # ═══ 门控温控自动激活（M_V已达稳定态且gate_std<0.03）═══
        if not self.gate_tau_active:
            mv_now = float(np.mean(np.linalg.norm(self.M_V, axis=1)))
            gate_std_now = float(np.std(gate))
            if mv_now >= 0.3 and gate_std_now < 0.03:
                self.gate_tau_active = True
                self.gate_sharper = DynamicGateSharper(
                    start_tau=0.2, target_tau=1.0, duration_topics=500000
                )
                print(f"  🔥 [GATE] M_V={mv_now:.3f}≥0.3 且 gate_σ={gate_std_now:.4f}<0.03 → 激活温控退火!", flush=True)
        
        # attn = softmax(top-8 scores * temp) 的梯度
        d_scores_raw = temp * attn * (d_attn - np.sum(attn * d_attn, axis=1, keepdims=True))
        # top-8 mask: 被遮挡的槽位梯度为0
        attn_mask = (scores_exp >= scores_topk).astype(np.float32)
        d_scores = d_scores_raw * attn_mask
        
        # scores = Q @ K_norm.T (K_norm = M_K / ||M_K||, 直通估计)
        self.M_K -= shared_lr * ((d_scores.T @ Q) + WEIGHT_DECAY * self.M_K)
        d_Q = d_scores @ K_norm
        
        # Q = h1_relu @ W_q
        self.W_q -= shared_lr * ((h1_relu.T @ d_Q) + WEIGHT_DECAY * self.W_q)
        d_h1_query = d_Q @ self.W_q.T
        
        # W_p 梯度（通过第一层ReLU反向，汇聚门控+直接+查询三条路径）
        grad_h1 = d_h1_direct + d_h1_gate + d_h1_query
        grad_h1_relu = grad_h1 * (h1 > 0).astype(np.float32)
        self.head.W_p -= shared_lr * ((X_masked.T @ grad_h1_relu) + WEIGHT_DECAY * self.head.W_p)
        
        # W_embed 梯度 (通过输入投影ReLU → W_p反向)
        grad_X_proj = grad_h1_relu @ self.head.W_p.T
        grad_X_proj_relu = grad_X_proj * (X_proj > 0).astype(np.float32)
        self.W_embed -= shared_lr * ((X_masked.T @ grad_X_proj_relu) + WEIGHT_DECAY * self.W_embed) * 0.3

        self.head.n_updates += 1
        self.head.total_loss += loss
        self.state["train_steps"] += batch_size
        self.state["total_loss"] += loss * batch_size

        return {
            "batch_size": batch_size,
            "recon_mse": recon_mse,
            "mem_mse": mem_mse,
            "word_bce": word_bce,
            "word_wce": word_bce_weighted,  # 加权BCE（监控）
            "contrastive": contrastive_loss,
            "h_var": h_var,
            "avg_loss": loss,
            "val_loss": val_mse,
            "avg_reward": float(np.mean(R)),
        }

    def _evolve(self):
        """自我进化 — 使用最近MSE作为适应度，加入var惩罚防止表征坍塌"""
        # 从最近N次训练的平均MSE计算适应度
        recent_mse = getattr(self, '_recent_mse', None)
        recent_var = getattr(self, '_recent_var', None)
        recent_word = getattr(self, '_recent_word_bce', None)
        
        # 基础fitness = 1/(1+avg_mse)
        if recent_mse is not None and len(recent_mse) > 0:
            avg_mse = sum(recent_mse) / len(recent_mse)
            fitness_base = 1.0 / (1.0 + avg_mse)
        else:
            fitness_base = 0.5
        
        # ═══ 表征坍塌检测：var惩罚 + word_bce陷阱检测 ═══
        collapse_penalty = 1.0
        collapse_flags = []
        
        # 1) 方差惩罚：var接近0 → 表征坍塌
        if recent_var is not None and len(recent_var) > 0:
            avg_var = sum(recent_var) / len(recent_var)
            if avg_var < 1e-6:  # 完全坍塌
                collapse_penalty *= 0.0
                collapse_flags.append("var=0完全坍塌")
            elif avg_var < 0.0002:  # 严重坍塌
                collapse_penalty *= min(1.0, avg_var / 0.0005)
                collapse_flags.append(f"var={avg_var:.4f}严重坍塌")
            elif avg_var < 0.0006:  # 低方差警告
                collapse_penalty *= min(1.0, avg_var / 0.001)
                collapse_flags.append(f"var={avg_var:.4f}偏低")
        
        # 2) word_bce陷阱检测：≈0.693 = 均匀分布退化
        if recent_word is not None and len(recent_word) > 0:
            avg_word = sum(recent_word) / len(recent_word)
            if abs(avg_word - 0.693) < 0.003:
                collapse_penalty *= 0.3  # 重度惩罚
                collapse_flags.append("word≈0.693均匀分布退化")
            elif abs(avg_word - 0.693) < 0.05:
                collapse_penalty *= 0.6
        
        # 应用惩罚
        fitness = round(fitness_base * collapse_penalty, 4)
        
        # 如果坍塌严重，直接归零
        if collapse_penalty < 0.1:
            fitness = 0.0
        
        self._recent_mse = []  # 重置，重新收集
        self._recent_var = []
        self._recent_word_bce = []
        self.state["evolutions"] += 1
        self.state["fitness"] = fitness
        if collapse_flags:
            logging.warning(f"🧬 进化: fitness={fitness} 坍塌标志: {'; '.join(collapse_flags)}")
        return fitness

    def _auto_evolve(self, result: dict):
        """自主进化 — 监控训练轨迹，自动调参
        
        监测指标：recon趋势、停滞检测、退化检测
        自动调整：对比权重、掩码比例、噪声水平、温度
        """
        from collections import deque
        if not hasattr(self, '_evo_history'):
            self._evo_history = deque(maxlen=50)
            self._evo_params = {
                'contrastive': CONTRASTIVE_WEIGHT,
                'mask': MASK_RATIO,
                'noise': INPUT_NOISE,
                'stagnation': 0,
                'best_recon': float('inf'),
                'best_batch': 0,
            }
        
        h = self._evo_history
        p = self._evo_params
        recon = result['recon_mse']
        h.append(recon)
        
        evolutions_made = []
        
        # ── 1. 检测新纪录 ──
        if recon < p['best_recon'] * 0.99:
            p['stagnation'] = 0
            p['best_recon'] = recon
            p['best_batch'] = self.state.get('train_steps', 0)
            return evolutions_made
        
        # ── 2. 停滞检测（最近15 batch 无改善） ──
        if len(h) >= 15:
            recent = list(h)[-15:]
            improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-10)
            if improvement < 0.01:  # 不足1%改善
                p['stagnation'] += 1
            else:
                p['stagnation'] = max(0, p['stagnation'] - 1)
        
        # ── 3. 退化检测（recon 突然升高） ──
        if len(h) >= 3:
            recent_3 = list(h)[-3:]
            if recent_3[-1] > recent_3[0] * 1.05:  # 升高超5%
                evolutions_made.append("⚠️ 检测到退化")
                # 降低学习率防止发散
                self.head.lr = max(0.001, self.head.lr * 0.8)
                evolutions_made.append(f"  lr→{self.head.lr:.5f}")
                return evolutions_made
        
        # ── 4. 停滞超过5次 → 执行进化 ──
        if p['stagnation'] >= 5:
            stag_count = p['stagnation']  # 先保存计数再清零
            p['stagnation'] = 0
            evolutions_made.append(f"🧬 停滞{stag_count}次，启动自动调参")
            
            var = result.get('h_var', 0.0005)
            
            # ═══ 4c: 表征坍塌/深度停滞检测 ═══
            word = result.get('word_bce', 1.0)
            recon = result.get('recon_mse', 0.001)
            # 完全坍塌：var≈0 或 word≈0.693+recon极低
            is_collapsed = (var < 1e-8) or (abs(word - 0.693) < 0.005 and recon < 0.0005)
            # 深度停滞：var<0.01且recon连续无改善（40K batch稀释陷阱）
            is_deep_stuck = (var < 0.01) and (p['stagnation'] >= 8)
            
            if is_collapsed:
                evolutions_made.append("🚨 检测到表征坍塌！执行权重扰动唤醒")
                pert_std = 0.05  # 扰动幅度5%
                # 扰动Embedding层（最重要：打破输入编码对称性）
                noise_e = np.random.randn(*self.W_embed.shape).astype(np.float32) * pert_std
                self.W_embed += noise_e
                evolutions_made.append(f"  W_embed: ∓{pert_std:.3f}噪声 (||n||={np.linalg.norm(noise_e)/np.sqrt(noise_e.size):.4f})")
                # 扰动投影层 W_p
                self.head.W_p += np.random.randn(*self.head.W_p.shape).astype(np.float32) * pert_std
                # 扰动记忆槽
                self.M_K += np.random.randn(*self.M_K.shape).astype(np.float32) * pert_std * 0.5
                self.M_K = self.M_K / (np.linalg.norm(self.M_K, axis=1, keepdims=True) + 1e-8)
                self.M_V += np.random.randn(*self.M_V.shape).astype(np.float32) * pert_std * 0.5
                # 扰动门控层
                self.W_gate += np.random.randn(*self.W_gate.shape).astype(np.float32) * pert_std
                self.b_gate += np.random.randn(*self.b_gate.shape).astype(np.float32) * pert_std * 0.3
                # 重置
                p['contrastive'] = min(5.0, CONTRASTIVE_WEIGHT * 2.0)
                p['noise'] = 0.15
                p['best_recon'] = float('inf')
                self._recent_mse = []
                if hasattr(self, '_evo_history'):
                    self._evo_history.clear()
                evolutions_made.append(f"  对比→{p['contrastive']:.1f} 噪声→{p['noise']:.2f} 全部参数扰动{pert_std}")
            
            elif is_deep_stuck:
                evolutions_made.append("⛰️ 深度停滞(var<0.01)，轻量扰动+符号梯度解除稀释陷阱")
                pert_std = 0.02  # 仅2%扰动
                # 仅扰动W_embed和门控层（不动记忆槽）
                self.W_embed += np.random.randn(*self.W_embed.shape).astype(np.float32) * pert_std
                self.W_gate += np.random.randn(*self.W_gate.shape).astype(np.float32) * pert_std * 0.5
                self.b_gate += np.random.randn(*self.b_gate.shape).astype(np.float32) * pert_std * 0.3
                p['stagnation'] = max(0, p['stagnation'] // 2)
                p['contrastive'] = min(5.0, p['contrastive'] * 1.5)
                evolutions_made.append(f"  W_embed+gate扰动{pert_std:.3f} 对比→{p['contrastive']:.1f}")
            
            # 4a: 对比损失调节
            var = result.get('h_var', 0.0005)
            if var < 0.0006:  # 提高触发阈值，解决var=0.0004死区
                p['contrastive'] = min(2.5, p['contrastive'] + 0.2)  # 上限2.5
                evolutions_made.append(f"  var={var:.4f}过低 → 对比{p['contrastive']:.1f}")
            elif var > 0.001:  # var足够高→降低对比，保护词表头
                p['contrastive'] = max(0.5, p['contrastive'] - 0.2)  # 下限0.5
                evolutions_made.append(f"  var={var:.4f}已够高 → 对比{p['contrastive']:.1f} 保护词表头")
                # 噪声注入
                if p['noise'] < 0.15:
                    old_noise = p['noise']
                    p['noise'] = min(0.3, p['noise'] + 0.05)
                    evolutions_made.append(f"  噪声: {old_noise}→{p['noise']:.2f}")
            
            # 更新全局变量
            import daemon_v3 as _self_mod
            _self_mod.CONTRASTIVE_WEIGHT = p['contrastive']
            _self_mod.INPUT_NOISE = p['noise']
            
            # 记录进化
            self.state['auto_evolutions'] = self.state.get('auto_evolutions', 0) + 1
        
        # ── 5. 记忆槽检测：每50个batch检查并替换闲置槽 ──
        batch_count = self.state.get('train_steps', 0) // BATCH_SIZE
        if (hasattr(self, '_slot_hits') and 
            self._last_slot_prune_batch > 0 and
            batch_count - self._last_slot_prune_batch >= 50 and
            batch_count > 10):
            usage_rate = self._slot_hits.astype(np.float32) / (batch_count - self._last_slot_prune_batch)
            idle_slots = np.where(usage_rate < 0.02)[0]  # 使用率<2%的槽
            if len(idle_slots) > 0:
                self._slot_hits = np.zeros_like(self._slot_hits)  # 重置计数
                self._last_slot_prune_batch = batch_count
                # 重新初始化闲置槽（保留活跃槽的学习内容）
                for sid in idle_slots:
                    self.M_K[sid] = np.random.randn(256).astype(np.float32)
                    self.M_K[sid] /= np.linalg.norm(self.M_K[sid]) + 1e-8
                    self.M_V[sid] = np.random.randn(256).astype(np.float32) * 0.01
                evolutions_made.append(f"  🗑️ 替换{len(idle_slots)}个闲置记忆槽({idle_slots.tolist()})")
        elif batch_count > 10 and self._last_slot_prune_batch == 0:
            # 首次到达10个batch，初始化计数基准
            self._last_slot_prune_batch = batch_count
            self._slot_hits = np.zeros_like(self._slot_hits)
        
        return evolutions_made

    def _save_knowledge(self, text: str):
        """保存知识到文件"""
        fname = f"{self.state['topics']:06d}_{text[:30].replace(' ','_').replace('/','_')}.txt"
        try:
            with open(os.path.join(KB_DIR, fname), "w", encoding="utf-8") as f:
                f.write(text[:5000])
            self.state["knowledge_files"] += 1
        except Exception:
            pass

    def _report(self) -> str:
        """生成状态报告"""
        uptime = (datetime.now() - datetime.fromisoformat(self.state["started"])).total_seconds()
        hours = uptime / 3600
        avg_loss = self.state["total_loss"] / max(self.state["train_steps"], 1)

        report = (
            f"\n{'='*60}\n"
            f"  🧠 NeuroFlow v3 状态报告 | {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{'='*60}\n"
            f"  ⏱️  运行时间:   {hours:.1f} 小时\n"
            f"  📖 已学主题:   {self.state['topics']}\n"
            f"  🔄 训练步数:   {self.state['train_steps']}\n"
            f"  📚 轮次:       {self.data_source.epoch}\n"
            f"  📊 LR:         {self.head.lr:.6f}\n"
            f"  🧬 进化次数:   {self.state['evolutions']}\n"
            f"  💪 适应度:     {self.state['fitness']}\n"
            f"  📊 平均损失:   {avg_loss:.6f}\n"
            f"  📚 知识文件:   {self.state['knowledge_files']}\n"
            f"  ❌ 错误数:     {self.state['errors']}\n"
            f"  📝 最近学习:   {self.state['last_activity'][:80]}\n"
            f"  📡 数据源:     {self.state.get('source_stats', {})}\n"
            f"  📂 本地KB:     {self.data_source.size()}\n"
            f"{'='*60}"
        )

        with open(REPORT_FILE, "w") as f:
            f.write(report)
        return report

    def run_forever(self):
        """主循环 — 批量训练模式，最大化硬件利用率"""
        ds = self.data_source.size()
        print(f"[{datetime.now():%H:%M:%S}] 🧠 NeuroFlow v3 守护进程启动 (ALL-IN-ONE MODE)")
        print(f"[{datetime.now():%H:%M:%S}] 📚 本地知识: 内置{ds['builtin']}条 + {ds['kb_files']}文件")
        print(f"[{datetime.now():%H:%M:%S}] ⚙️  batch={BATCH_SIZE} | workers={PARALLEL_WORKERS} | dims={TEXT_DIM}→{HIDDEN_DIM}→{HIDDEN2_DIM}→{OUTPUT_DIM}+{MEM_DIM}")
        print(f"[{datetime.now():%H:%M:%S}] 🔄 架构: W_embed→W_p→ReLU→GatedMemBank(32槽×256→top6)→SAE(top65/512,12.7%)→[W_d+W_m+W_v+W_gen({VOCAB_SIZE}词)]")

        batch_count = 0
        while True:
            try:
                # 批量学习（BATCH_SIZE条，一次前传+一次批量梯度更新）
                result = self._learn_batch()
                self.state["topics"] += result["batch_size"]
                batch_count += 1

                # 进化和保存（基于总条数）
                topics = self.state["topics"]

                # 进化
                if topics % EVOLVE_INTERVAL < result["batch_size"] and topics > 0:
                    fitness = self._evolve()
                    avg_loss = self.state["total_loss"] / max(self.state["train_steps"], 1)
                    print(f"  🧬 进化 #{self.state['evolutions']}: "
                          f"fitness={fitness} loss={avg_loss:.4f} "
                          f"topics={topics} batch={batch_count}", flush=True)

                # 保存
                if topics % SAVE_EVERY < result["batch_size"]:
                    self._save_weights()
                    self._save_state()

                # 日志（每BATCH_LOG_EVERY个batch输出一行）
                if batch_count % BATCH_LOG_EVERY == 0:
                    avg_loss = self.state["total_loss"] / max(self.state["train_steps"], 1)
                    fps = result["batch_size"] / max(0.01, time.time() - getattr(self, '_last_log', time.time()))
                    self._last_log = time.time()
                    bar_len = min(30, int(result["avg_reward"] * 30))
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    print(f"  [{topics:7d}] 📦 batch#{batch_count} e{self.data_source.epoch} "
                          f"recon={result['recon_mse']:.6f} word={result.get('word_bce',0):.4f} "
                          f"wce={result.get('word_wce',0):.2f} "
                          f"var={result.get('h_var',0):.4f} k={getattr(self,'_sae_k',65)} "
                          f"fit={self.state['fitness']} | {bar}", flush=True)

                # 定时状态汇报
                now = time.time()
                if now - self.last_status >= STATUS_INTERVAL:
                    print(flush=True)
                    report = self._report()
                    print(report, flush=True)
                    self._save_state()
                    self.last_status = now
                
                # 每2个batch训练一次独立词表头（原为5，加速收敛）
                if batch_count % 2 == 0 and batch_count > 0:
                    v_result = self._train_vocab_separately()
                    if v_result["vocab_acc"] > 0.01:
                        print(f"  📝 词表头训练: loss={v_result['vocab_loss']:.4f} "
                              f"top5命中率={v_result['vocab_acc']:.2%}", flush=True)

                # 自主进化：每个batch后监控轨迹，自动调参
                evo_actions = self._auto_evolve(result)
                for action in evo_actions:
                    print(f"  {action}", flush=True)

            except KeyboardInterrupt:
                print(f"\n[{datetime.now():%H:%M:%S}] ⏸️  保存状态并退出...")
                self._save_state()
                self._save_weights()
                break
            except Exception as e:
                self.state["errors"] += 1
                print(f"\n  ⚠️ [{datetime.now():%H:%M:%S}] batch#{batch_count} {e}", flush=True)
                time.sleep(5)


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    daemon = NeuroFlowDaemonV3()
    daemon.run_forever()
