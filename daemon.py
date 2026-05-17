"""
NeuroFlow 自主进化守护进程
===========================
持续从互联网学习知识 → 自主反思 → 进化迭代 → 状态汇报

部署路径: D:/neuroflow-model/
状态日志: D:/neuroflow-model/daemon_state.json
知识库:   D:/neuroflow-model/knowledge_base/

运行方式:
  python3 daemon.py           # 前台运行
  nohup python3 daemon.py &   # 后台守护
"""

import sys, os, time, json, random, urllib.request, urllib.parse
import numpy as np
from datetime import datetime
from collections import deque

sys.path.insert(0, "/mnt/d/neuroflow-model")

from neuroflow._core import create_multimodal, NeuroFlowLite
from neuroflow.cognition import ReasoningLoop, SelfEvolution, AutonomousAgent
from neuroflow.decoder import TextDecoder


# ============================================================
# 配置
# ============================================================
DEPLOY_PATH = "/mnt/d/neuroflow-model"
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_state.json")
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")
STATUS_INTERVAL = 600  # 10分钟 = 600秒
LEARNING_INTERVAL = 30  # 每30秒学习一次

os.makedirs(KB_DIR, exist_ok=True)


# ============================================================
# 状态管理器
# ============================================================

class DaemonState:
    """守护进程状态追踪"""

    def __init__(self):
        self.start_time = datetime.now().isoformat()
        self.topics_learned = 0
        self.evolution_cycles = 0
        self.current_fitness = 0.0
        self.knowledge_count = 0
        self.errors = 0
        self.last_activity = ""
        self.recent_topics = deque(maxlen=20)
        self.fitness_history = []
        self.uptime_seconds = 0

    def log_activity(self, activity: str):
        self.last_activity = activity

    def record_topic(self, topic: str):
        self.topics_learned += 1
        self.recent_topics.append(topic)

    def snapshot(self) -> dict:
        self.uptime_seconds = (datetime.now() - datetime.fromisoformat(self.start_time)).total_seconds()
        return {
            "started": self.start_time,
            "uptime_hours": round(self.uptime_seconds / 3600, 2),
            "topics_learned": self.topics_learned,
            "evolution_cycles": self.evolution_cycles,
            "fitness": round(self.current_fitness, 4),
            "knowledge_count": self.knowledge_count,
            "errors": self.errors,
            "last_activity": self.last_activity,
            "recent_topics": list(self.recent_topics)[-10:],
            "fitness_trend": self.fitness_history[-20:] if self.fitness_history else [],
            "updated": datetime.now().isoformat(),
        }

    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.snapshot(), f, indent=2, ensure_ascii=False)


# ============================================================
# 互联网知识采集器
# ============================================================

class InternetLearner:
    """
    从互联网自主学习知识。
    支持: Wikipedia, 新闻, 通用网页
    """

    # 初始学习主题（逐步扩展）
    TOPIC_POOL = [
        # 科学
        "artificial intelligence basics",
        "neural network architecture",
        "quantum computing principles",
        "theory of relativity explained",
        "evolution natural selection",
        "DNA structure function",
        "climate change science",
        "ocean currents world",
        "plate tectonics earth",
        "black hole formation",
        # 哲学与认知
        "consciousness philosophy mind",
        "free will determinism debate",
        "ethics artificial intelligence",
        "philosophy of science",
        "epistemology knowledge theory",
        # 技术与工程
        "computer architecture basics",
        "operating system design",
        "database management systems",
        "cryptography principles",
        "distributed systems design",
        # 人文
        "world history timeline",
        "ancient civilizations",
        "renaissance art science",
        "industrial revolution impact",
        "democracy origins",
    ]

    def __init__(self):
        self.learned = set()
        self.topic_index = 0

    def get_next_topic(self) -> str:
        """获取下一个学习主题"""
        if self.topic_index >= len(self.TOPIC_POOL):
            self.topic_index = 0
        topic = self.TOPIC_POOL[self.topic_index]
        self.topic_index += 1
        return topic

    def fetch_knowledge(self, topic: str) -> str:
        """
        从网络获取知识内容。
        优先 Wikipedia API, 回退到通用搜索。
        """
        # 方式1: Wikipedia API
        try:
            encoded = urllib.parse.quote(topic)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "NeuroFlow/3.2 (Learning Agent; contact@neuroflow.ai)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                extract = data.get("extract", "")
                title = data.get("title", topic)
                if len(extract) > 100:
                    return f"[Wikipedia] {title}: {extract[:2000]}"
        except Exception:
            pass

        # 方式2: DuckDuckGo Instant Answer API (free, no key)
        try:
            encoded = urllib.parse.quote(topic)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "NeuroFlow/3.2"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                abstract = data.get("Abstract", "")
                if len(abstract) > 50:
                    return f"[DuckDuckGo] {topic}: {abstract[:2000]}"
        except Exception:
            pass

        return f"[Local] {topic}: Fundamental knowledge about {topic}."

    def learn_topic(self, topic: str) -> tuple:
        """学习一个主题，返回 (topic, content, success)"""
        content = self.fetch_knowledge(topic)
        success = not content.startswith("[Local]")
        self.learned.add(topic)
        return topic, content, success


# ============================================================
# 知识编码器
# ============================================================

class InternetKnowledgeEncoder:
    """将网络文本编码为 NeuroFlow 可学习的特征向量"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)

        for i, word in enumerate(words[:500]):
            h = hash(word) % (2**31)
            for j in range(8):
                idx = (h + j * 2654435761) % self.dim
                vec[int(idx)] += 0.02 / max(len(words) / 50, 1)

        # 添加文本结构信号
        vec += np.sin(np.linspace(0, np.pi * len(words) / 20, self.dim)).astype(np.float32) * 0.1
        return vec / (np.linalg.norm(vec) + 1e-8)


# ============================================================
# 自主进化守护进程
# ============================================================

class NeuroFlowDaemon:
    """
    NeuroFlow 自主进化守护进程。
    
    循环:
      1. 从互联网获取新知识
      2. 编码 → 推理 → 学习
      3. 定期自我反思
      4. 定期进化
      5. 保存状态
    """

    def __init__(self):
        print(f"[{datetime.now():%H:%M:%S}] 🧠 NeuroFlow Daemon v3.2 启动中...")

        # 模型
        self.model = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=True)

        # 智能体
        self.agent = AutonomousAgent(self.model, name="NF-Daemon")

        # 模块
        self.learner = InternetLearner()
        self.encoder = InternetKnowledgeEncoder(dim=512)

        # 状态
        self.state = DaemonState()
        self._load_state()

        # 统计
        self.cycle_count = 0
        self.last_evolution = 0
        self.last_status = time.time()

        print(f"[{datetime.now():%H:%M:%S}] ✅ 就绪 — 知识库: {self.state.knowledge_count}条")
        print(f"[{datetime.now():%H:%M:%S}] 🌐 开始自主学习...")

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    saved = json.load(f)
                self.state.topics_learned = saved.get("topics_learned", 0)
                self.state.evolution_cycles = saved.get("evolution_cycles", 0)
                self.state.knowledge_count = saved.get("knowledge_count", 0)
                self.state.fitness_history = saved.get("fitness_trend", [])
            except Exception:
                pass

    def _encode_and_learn(self, content: str) -> float:
        """编码知识并让智能体学习"""
        x = self.encoder.encode(content).reshape(1, -1)

        # 推理
        trace = self.agent.reasoner.reason(x)

        # 评估学习效果
        confidence = trace.thoughts[-1].confidence if trace.thoughts else 0.5
        stability = self._calc_stability(trace)
        reward = (confidence + stability) / 2

        # 学习
        self.agent.learn_from_feedback(x, reward)
        return reward

    def _calc_stability(self, trace) -> float:
        if not trace or len(trace.thoughts) < 2:
            return 0.5
        confs = [t.confidence for t in trace.thoughts]
        return 1.0 - min(float(np.std(confs)), 0.5)

    def _evolve_if_ready(self):
        """定期进化"""
        if self.state.topics_learned - self.last_evolution >= 30:
            self.state.log_activity("Evolution cycle starting")
            self.agent.evolution.evolve(generations=15, verbose=False)
            self.agent.evolution.consolidate()
            self.state.evolution_cycles += 1
            self.last_evolution = self.state.topics_learned

            fitness = self.agent.evolution.best_fitness
            self.state.current_fitness = fitness
            self.state.fitness_history.append(fitness)
            self.state.log_activity(f"Evolved: fitness={fitness:.4f}")

    def _save_knowledge(self, topic: str, content: str):
        """保存知识到磁盘"""
        fname = f"{self.state.topics_learned:06d}_{topic[:30].replace(' ','_').replace('/','_')}.txt"
        fpath = os.path.join(KB_DIR, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content[:5000])
            self.state.knowledge_count += 1
        except Exception:
            pass

    def run_cycle(self):
        """单次学习循环"""
        try:
            # 1. 获取新知识
            topic = self.learner.get_next_topic()
            topic_name, content, from_web = self.learner.learn_topic(topic)

            # 2. 学习
            reward = self._encode_and_learn(content)
            self.state.record_topic(topic_name)
            self.state.log_activity(f"Learned: {topic_name[:50]} (reward={reward:.2f})")

            # 3. 保存知识
            self._save_knowledge(topic_name, content)

            # 4. 检查进化
            self._evolve_if_ready()

            self.cycle_count += 1

            # 打印（前几次详细）
            source = "🌐" if from_web else "📚"
            print(f"  [{self.cycle_count:4d}] {source} {topic_name[:40]:40s} "
                  f"reward={reward:.2f} | fitness={self.state.current_fitness:.4f}")

        except Exception as e:
            self.state.errors += 1
            print(f"  ⚠️ Error: {e}")

    def status_report(self) -> str:
        """生成状态报告"""
        snap = self.state.snapshot()
        return (
            f"\n{'='*60}\n"
            f"  📊 NeuroFlow 状态报告 | {datetime.now():%H:%M:%S}\n"
            f"{'='*60}\n"
            f"  ⏱️ 运行时间:   {snap['uptime_hours']:.1f} 小时\n"
            f"  📖 已学主题:   {snap['topics_learned']}\n"
            f"  🧬 进化次数:   {snap['evolution_cycles']}\n"
            f"  💪 适应度:     {snap['fitness']}\n"
            f"  📚 知识库:     {snap['knowledge_count']} 条\n"
            f"  ❌ 错误数:     {snap['errors']}\n"
            f"  📝 最近学习:   {snap['last_activity'][:80]}\n"
            f"  📈 进化趋势:   {snap['fitness_trend']}\n"
            f"{'='*60}\n"
        )

    def run_forever(self):
        """主循环：永远运行"""
        print(f"[{datetime.now():%H:%M:%S}] 🚀 NeuroFlow 自主进化守护进程启动")
        print(f"[{datetime.now():%H:%M:%S}] 📋 状态间隔: {STATUS_INTERVAL}s | 学习间隔: {LEARNING_INTERVAL}s")
        print()

        while True:
            try:
                self.run_cycle()
                self.state.save()

                # 定时状态汇报
                if time.time() - self.last_status >= STATUS_INTERVAL:
                    report = self.status_report()
                    print(report, flush=True)

                    # 写报告文件
                    report_file = os.path.join(DEPLOY_PATH, "status_report.txt")
                    with open(report_file, "w") as f:
                        f.write(report)
                    print(f"  📄 报告已保存: {report_file}", flush=True)

                    self.last_status = time.time()

                time.sleep(LEARNING_INTERVAL)

            except KeyboardInterrupt:
                print(f"\n[{datetime.now():%H:%M:%S}] ⏸️ 收到停止信号，保存状态...")
                self.state.save()
                print(f"[{datetime.now():%H:%M:%S}] ✅ 状态已保存。再见！")
                break
            except Exception as e:
                self.state.errors += 1
                print(f"  ❌ 主循环异常: {e}")
                time.sleep(5)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    daemon = NeuroFlowDaemon()
    daemon.run_forever()
