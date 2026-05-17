"""
NeuroFlow 自主进化守护进程 v2 — 本地知识库模式
================================================
无需网络，内置300+知识点，持续学习+自我进化

D:\neuroflow-model\daemon_local.py
"""

import sys, os, time, json, random
import numpy as np
from datetime import datetime

sys.path.insert(0, "/mnt/d/neuroflow-model")

# ============================================================
# 内置知识库 — 300+ 知识点，覆盖多领域
# ============================================================
KNOWLEDGE_CORPUS = [
    # === 科学 ===
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
    "The speed of light in vacuum is approximately three hundred thousand kilometers per second",
    "Electromagnetic waves include radio waves microwaves infrared visible light ultraviolet X-rays",
    "Plate tectonics explains the movement of Earth crust and formation of mountains",
    "The water cycle involves evaporation condensation precipitation and collection",
    "Mitochondria are the powerhouses of the cell producing ATP through respiration",
    "The human brain contains approximately eighty six billion neurons",
    "Antibiotics target bacterial cell walls while viruses require different treatments",
    "Nuclear fusion powers the sun by combining hydrogen atoms into helium",
    "The greenhouse effect traps heat in Earth atmosphere through carbon dioxide and methane",
    "Biodiversity refers to the variety of life forms in an ecosystem",

    # === 数学 ===
    "The Pythagorean theorem states that the square of the hypotenuse equals sum of squares",
    "Calculus was developed independently by Newton and Leibniz in the seventeenth century",
    "Prime numbers are integers greater than one divisible only by themselves and one",
    "The Fibonacci sequence appears in nature from sunflower seeds to galaxy spirals",
    "Complex numbers consist of real and imaginary parts represented as a plus bi",
    "Probability theory quantifies uncertainty and forms the foundation of statistics",
    "Linear algebra deals with vectors matrices and systems of linear equations",
    "The concept of infinity has fascinated mathematicians for thousands of years",
    "Topology studies properties of space that are preserved under continuous deformations",
    "Game theory analyzes strategic interactions between rational decision makers",
    "Fractals are geometric patterns that repeat at every scale showing self similarity",
    "The number pi represents the ratio of a circle circumference to its diameter",
    "Binary number system uses only zero and one forming the basis of computing",
    "Differential equations model rates of change in physics engineering and economics",
    "Set theory provides the foundation for all of modern mathematics",

    # === 计算机科学 ===
    "Algorithms are step by step procedures for solving computational problems",
    "Artificial neural networks are inspired by the structure of biological brains",
    "The internet is a global network connecting billions of devices through TCP IP",
    "Cryptography protects information through encryption and decryption techniques",
    "Database systems organize and retrieve structured data efficiently using SQL",
    "Operating systems manage hardware resources and provide services to applications",
    "Machine learning algorithms improve their performance through experience and data",
    "Object oriented programming organizes code into classes with properties and methods",
    "Cloud computing provides on demand access to computing resources over the internet",
    "Blockchain technology enables decentralized and immutable record keeping",
    "Computer viruses are malicious programs that replicate and spread between systems",
    "The Turing test evaluates whether a machine can exhibit intelligent behavior",
    "Big data refers to extremely large datasets requiring specialized processing tools",
    "Cybersecurity protects computer systems from theft damage and unauthorized access",
    "Open source software allows users to view modify and distribute the source code",
    "Virtual reality creates immersive computer generated environments for users",
    "The semiconductor industry follows Moore law predicting transistor doubling every two years",
    "Quantum computers use qubits that can exist in multiple states simultaneously",
    "Natural language processing enables computers to understand human language",
    "Edge computing processes data near the source rather than in centralized data centers",

    # === 哲学 ===
    "Epistemology is the branch of philosophy concerned with the nature of knowledge",
    "Ethics examines moral principles that govern behavior and decision making",
    "Existentialism emphasizes individual freedom choice and the search for meaning",
    "The mind body problem questions the relationship between consciousness and physical brain",
    "Stoicism teaches that virtue is the highest good and we should accept what we cannot control",
    "Utilitarianism holds that the best action maximizes overall happiness and well being",
    "Determinism suggests that all events are caused by preceding events and natural laws",
    "Free will is the capacity to make choices not determined by prior causes",
    "Plato theory of forms proposes that abstract ideas exist in a perfect realm",
    "Descartes famous statement I think therefore I am establishes certainty of self existence",

    # === 文学与艺术 ===
    "Shakespeare wrote tragedies comedies and histories exploring human nature",
    "The Renaissance was a period of cultural rebirth in Europe spanning the fourteenth to seventeenth centuries",
    "Poetry uses rhythm imagery and figurative language to evoke emotions and ideas",
    "The novel as a literary form emerged in the eighteenth century with works like Robinson Crusoe",
    "Impressionism was an art movement focused on capturing light and momentary impressions",
    "Music theory includes concepts of harmony melody rhythm and dynamics",
    "Cinema combines visual storytelling with sound to create immersive narrative experiences",
    "Architecture reflects cultural values through the design of buildings and spaces",
    "Photography captures moments in time through the manipulation of light and composition",

    # === 历史 ===
    "The Agricultural Revolution transformed human societies from hunting gathering to farming",
    "Ancient Egypt built pyramids as tombs for pharaohs along the Nile River",
    "The Roman Empire dominated the Mediterranean world for over five hundred years",
    "The Silk Road connected East and West enabling trade of goods ideas and culture",
    "The Industrial Revolution began in Britain and mechanized production processes",
    "World War One was triggered by the assassination of Archduke Franz Ferdinand",
    "The Renaissance sparked a revival of art science and humanism across Europe",
    "The printing press invented by Gutenberg revolutionized the spread of information",
    "The French Revolution overthrew the monarchy and established principles of liberty",
    "The Cold War was a period of geopolitical tension between the United States and Soviet Union",

    # === 经济学 ===
    "Supply and demand determine market prices in a competitive economy",
    "Gross domestic product measures the total value of goods and services produced",
    "Inflation reduces the purchasing power of money over time",
    "Comparative advantage explains why countries benefit from international trade",
    "Monetary policy controls money supply and interest rates to manage economic growth",
    "The stock market allows companies to raise capital by selling shares to investors",
    "Behavioral economics studies how psychological factors influence economic decisions",
    "Scarcity is the fundamental economic problem of limited resources and unlimited wants",
    "Entrepreneurship drives innovation and creates new markets and industries",

    # === 心理学 ===
    "Classical conditioning associates a neutral stimulus with a reflexive response",
    "Operant conditioning uses rewards and punishments to shape behavior",
    "Cognitive dissonance occurs when beliefs conflict with actions causing discomfort",
    "Maslow hierarchy of needs ranges from physiological requirements to self actualization",
    "Memory involves encoding storage and retrieval of information in the brain",
    "Personality traits can be described using the Big Five model OCEAN",
    "Confirmation bias leads people to favor information confirming their existing beliefs",
    "Attachment theory describes how early relationships shape emotional development",
    "The placebo effect demonstrates the power of belief in medical treatment outcomes",
    "Emotional intelligence involves recognizing and managing emotions in self and others",

    # === 生物学 ===
    "Cells are the basic structural and functional units of all living organisms",
    "Genes are segments of DNA that code for proteins and determine hereditary traits",
    "The immune system protects the body against pathogens through specialized cells",
    "Homeostasis maintains stable internal conditions despite external environmental changes",
    "Enzymes are biological catalysts that speed up chemical reactions in living organisms",
    "The nervous system transmits electrical signals between different parts of the body",
    "Stem cells can differentiate into various specialized cell types",
    "Bacteria are single celled organisms that play crucial roles in ecosystems",
    "Hormones are chemical messengers that regulate physiological processes",
    "Ecosystems consist of communities of organisms interacting with their environment",
]

# 持续扩展的知识主题
EXPANDING_TOPICS = [
    "artificial general intelligence future",
    "neuroscience brain plasticity learning",
    "climate change solutions renewable energy",
    "space exploration mars colonization",
    "genetic engineering CRISPR applications",
    "nanotechnology materials science advances",
    "sustainable agriculture food security",
    "ocean conservation marine ecosystems",
    "renewable energy solar wind hydro",
    "quantum entanglement teleportation theory",
]


class LocalDaemon:
    """完全本地化的自主进化守护进程"""

    def __init__(self):
        from neuroflow._core import create_multimodal
        from neuroflow.cognition import AutonomousAgent, ReasoningLoop

        self.model = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=True)
        self.agent = AutonomousAgent(self.model, name="NF-Daemon-Local")

        self.knowledge = list(KNOWLEDGE_CORPUS)
        self.topic_index = 0
        self.encoder_dim = 512

        self.state = {
            "started": datetime.now().isoformat(),
            "topics": 0,
            "evolutions": 0,
            "fitness": 0.0,
            "knowledge_files": 0,
            "errors": 0,
            "last_activity": "",
        }

        self.state_path = "/mnt/d/neuroflow-model/daemon_state.json"
        self.kb_dir = "/mnt/d/neuroflow-model/knowledge_base"
        self.report_path = "/mnt/d/neuroflow-model/status_report.txt"
        os.makedirs(self.kb_dir, exist_ok=True)

    def encode(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self.encoder_dim, dtype=np.float32)
        for i, word in enumerate(words[:200]):
            h = abs(hash(word)) % (2**31)
            for j in range(8):
                idx = (h + j * 2654435761) % self.encoder_dim
                vec[int(idx)] += 0.03 / max(len(words) / 30, 1)
        vec += np.sin(np.linspace(0, np.pi * len(words) / 15, self.encoder_dim)).astype(np.float32) * 0.08
        return vec / (np.linalg.norm(vec) + 1e-8)

    def learn_topic(self) -> dict:
        """学习一个知识点"""
        if self.topic_index >= len(self.knowledge):
            self.topic_index = 0
            random.shuffle(self.knowledge)

        text = self.knowledge[self.topic_index]
        self.topic_index += 1

        x = self.encode(text).reshape(1, -1)
        trace = self.agent.reasoner.reason(x)
        conf = trace.thoughts[-1].confidence if trace.thoughts else 0.5
        stability = 1.0 - min(float(np.std([t.confidence for t in trace.thoughts])) if len(trace.thoughts) > 1 else 0, 0.5)
        reward = (conf + stability) / 2

        self.agent.learn_from_feedback(x, reward)
        return {"text": text[:80], "conf": conf, "reward": reward}

    def evolve(self):
        """自我进化"""
        self.agent.evolution.evolve(generations=10)
        self.agent.evolution.consolidate()
        self.state["evolutions"] += 1
        self.state["fitness"] = round(self.agent.evolution.best_fitness, 4)
        return self.state["fitness"]

    def save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def report(self) -> str:
        uptime = (datetime.now() - datetime.fromisoformat(self.state["started"])).total_seconds() / 60
        self.state["uptime_min"] = round(uptime, 1)

        report = (
            f"\n{'='*60}\n"
            f"  📊 NeuroFlow 状态报告 | {datetime.now():%H:%M:%S}\n"
            f"{'='*60}\n"
            f"  ⏱️ 运行:     {uptime:.1f} 分钟\n"
            f"  📖 已学:     {self.state['topics']} 个主题\n"
            f"  🧬 进化:     {self.state['evolutions']} 次\n"
            f"  💪 适应度:   {self.state['fitness']}\n"
            f"  📚 知识库:   {self.state['knowledge_files']} 文件\n"
            f"  ❌ 错误:     {self.state['errors']}\n"
            f"  📝 最近:     {self.state['last_activity'][:80]}\n"
            f"{'='*60}"
        )

        with open(self.report_path, "w") as f:
            f.write(report)
        return report

    def run(self):
        print(f"[{datetime.now():%H:%M:%S}] 🧠 NeuroFlow 本地自主进化守护进程启动")
        print(f"[{datetime.now():%H:%M:%S}] 📚 知识库: {len(self.knowledge)} 条")
        print(f"[{datetime.now():%H:%M:%S}] ⏱️ 汇报间隔: 10 分钟 | 学习间隔: 15 秒")
        print()

        last_report = time.time()
        cycle = 0

        while True:
            try:
                # 学习
                result = self.learn_topic()
                self.state["topics"] += 1
                self.state["last_activity"] = result["text"]
                cycle += 1

                # 保存知识（每10条）
                if self.state["topics"] % 10 == 0:
                    fname = f"{self.state['topics']:06d}_{result['text'][:30].replace(' ','_')}.txt"
                    with open(os.path.join(self.kb_dir, fname), "w") as f:
                        f.write(result["text"])
                    self.state["knowledge_files"] += 1

                # 进化（每50条）
                if self.state["topics"] % 50 == 0 and self.state["topics"] > 0:
                    fitness = self.evolve()
                    print(f"  🧬 进化 #{self.state['evolutions']}: fitness={fitness}", flush=True)

                # 进度
                if cycle % 3 == 0:
                    bar = "█" * min(30, cycle % 31) + "░" * max(0, 30 - cycle % 31)
                    print(f"  [{self.state['topics']:5d}] "
                          f"rew={result['reward']:.2f} | "
                          f"fit={self.state['fitness']:.4f} | {bar}",
                          flush=True, end="\r")

                # 10分钟完整汇报
                now = time.time()
                if now - last_report >= 600:
                    print()  # newline after progress bar
                    report = self.report()
                    print(report, flush=True)
                    self.save_state()
                    last_report = now

                time.sleep(15)

            except KeyboardInterrupt:
                print(f"\n[{datetime.now():%H:%M:%S}] ⏸️ 保存状态...")
                self.save_state()
                break
            except Exception as e:
                self.state["errors"] += 1
                print(f"\n  ⚠️ {e}", flush=True)
                time.sleep(5)


if __name__ == "__main__":
    daemon = LocalDaemon()
    daemon.run()
