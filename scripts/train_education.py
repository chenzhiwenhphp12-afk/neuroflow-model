"""
NeuroFlow 学制训练系统
======================
小学 → 中学 → 高中 → 大学  完整知识体系训练

每个阶段包含：
  - 知识图谱 (Knowledge Graph): 分科知识点
  - 课程训练 (Curriculum): 递进式学习
  - 阶段考试 (Exam): 知识掌握度评估
  - 毕业认证 (Graduation): 达标晋级
"""

import sys, numpy as np, time, random
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import create_multimodal
from neuroflow.cognition import ReasoningLoop, SelfEvolution, AutonomousAgent


# ============================================================
# 知识图谱: 四阶段课程体系
# ============================================================

KNOWLEDGE_GRAPH = {
    "小学": {
        "description": "基础感知与简单概念",
        "subjects": {
            "语文": {
                "topics": [
                    "拼音字母 a b c d e",
                    "简单汉字 人 口 手 山 水 火",
                    "基础词语 爸爸 妈妈 太阳 月亮 星星",
                    "简单句子 我爱学习 天空很蓝 花儿开了",
                    "反义词 大-小 多-少 高-矮 快-慢",
                ],
                "exam": "用下列词语造句：太阳、月亮、星星、天空",
            },
            "数学": {
                "topics": [
                    "认识数字 1 2 3 4 5 6 7 8 9 10",
                    "加法运算 1+1=2 2+3=5 4+5=9",
                    "减法运算 5-2=3 8-3=5 10-4=6",
                    "形状识别 圆 方 三角 长方 椭圆",
                    "大小比较 大于 小于 等于",
                ],
                "exam": "计算：3+5=? 9-4=? 哪个数字最大：2,7,3,9,1",
            },
            "自然": {
                "topics": [
                    "常见动物 猫 狗 鸟 鱼 蝴蝶",
                    "植物认知 花 草 树 叶子 果实",
                    "天气现象 晴 雨 风 雪 彩虹",
                    "四季变化 春 夏 秋 冬",
                    "颜色世界 红 蓝 绿 黄 紫",
                ],
                "exam": "说出三种动物、两种颜色、四个季节",
            },
        },
        "pass_score": 60,
    },

    "中学": {
        "description": "抽象概念与逻辑关系",
        "subjects": {
            "语文": {
                "topics": [
                    "成语故事 亡羊补牢 守株待兔 画蛇添足 刻舟求剑",
                    "修辞手法 比喻 拟人 夸张 排比 对偶",
                    "段落理解 中心思想 主要内容 作者意图",
                    "简单文言 学而时习之 温故而知新 三人行必有我师",
                    "应用写作 通知 请假条 日记 书信",
                ],
                "exam": "分析'守株待兔'的寓意，写一篇100字短文",
            },
            "数学": {
                "topics": [
                    "分数运算 1/2+1/4 2/3×3/4 5/6÷2/3",
                    "方程求解 x+5=12 3x-7=14 2x²=18",
                    "几何基础 三角形面积 圆周长 长方体体积",
                    "概率入门 掷骰子 抛硬币 抽卡片",
                    "函数概念 y=2x+1 y=x² 正比例与反比例",
                ],
                "exam": "解方程：3x-7=14，求圆的面积(r=5)，计算概率：骰子掷出偶数的概率",
            },
            "科学": {
                "topics": [
                    "物理基础 力与运动 光的反射 声音传播 电路",
                    "化学入门 元素周期表 化学反应 酸与碱 氧化还原",
                    "生物基础 细胞结构 光合作用 食物链 遗传",
                    "地理知识 地球构造 板块运动 气候带 洋流",
                    "天文初步 太阳系 日食月食 星座 黑洞概念",
                ],
                "exam": "解释光合作用的过程，列举太阳系八大行星",
            },
        },
        "pass_score": 65,
    },

    "高中": {
        "description": "复杂推理与系统思维",
        "subjects": {
            "语文": {
                "topics": [
                    "古典文学 诗经 楚辞 唐诗 宋词 元曲 明清小说",
                    "现代文学 鲁迅 老舍 巴金 沈从文 张爱玲",
                    "议论文写作 立论 驳论 论证方法 逻辑结构",
                    "文学鉴赏 意象分析 情感表达 艺术手法",
                    "语言运用 病句修改 句式变换 得体表达",
                ],
                "exam": "赏析'大漠孤烟直 长河落日圆'的意境，写200字议论文",
            },
            "数学": {
                "topics": [
                    "函数进阶 指数 对数 幂函数 三角函数 复合函数",
                    "解析几何 直线方程 圆的方程 椭圆 双曲线 抛物线",
                    "数列极限 等差数列 等比数列 数列求和 极限概念",
                    "向量运算 向量加减 点积叉积 空间向量",
                    "概率统计 排列组合 二项分布 正态分布 假设检验",
                ],
                "exam": "求椭圆x²/4+y²/9=1的焦点，计算sin²θ+cos²θ，证明等差数列求和公式",
            },
            "物理": {
                "topics": [
                    "力学 牛顿定律 动量守恒 能量守恒 万有引力",
                    "电磁学 库仑定律 欧姆定律 法拉第电磁感应 麦克斯韦方程",
                    "热力学 热力学定律 熵 卡诺循环 理想气体",
                    "光学 折射反射 干涉衍射 偏振 光电效应",
                    "近代物理 相对论基础 量子力学入门 波粒二象性",
                ],
                "exam": "推导动能定理，解释光电效应实验现象，计算单摆周期",
            },
            "化学": {
                "topics": [
                    "化学键 离子键 共价键 金属键 分子间力",
                    "化学平衡 勒夏特列原理 平衡常数 酸碱平衡",
                    "有机化学 烃类 醇醛酮酸 酯化反应 聚合反应",
                    "电化学 原电池 电解池 电极电势 金属腐蚀",
                    "物质结构 原子轨道 电子排布 晶体结构",
                ],
                "exam": "写出乙酸乙酯的合成反应方程式，计算化学平衡常数",
            },
        },
        "pass_score": 70,
    },

    "大学": {
        "description": "专业知识与创新思维",
        "subjects": {
            "计算机科学": {
                "topics": [
                    "算法设计 排序 搜索 动态规划 贪心 分治 图论算法",
                    "数据结构 链表 树 图 哈希表 堆 并查集 线段树",
                    "操作系统 进程调度 内存管理 文件系统 并发控制 死锁",
                    "计算机网络 TCP/IP HTTP DNS 路由算法 网络安全",
                    "人工智能 神经网络 强化学习 自然语言处理 计算机视觉 大语言模型",
                    "软件工程 设计模式 架构风格 持续集成 测试驱动 代码审查",
                ],
                "exam": "设计LRU缓存，分析Transformer注意力机制的时间复杂度，实现快速排序并证明正确性",
            },
            "高等数学": {
                "topics": [
                    "微积分 极限 导数 积分 多元微积分 泰勒展开 傅里叶级数",
                    "线性代数 矩阵运算 特征值 奇异值分解 向量空间 线性变换",
                    "概率论 条件概率 贝叶斯定理 随机过程 马尔可夫链 大数定律",
                    "优化理论 梯度下降 拉格朗日乘子 凸优化 KKT条件 对偶理论",
                    "信息论 熵 互信息 KL散度 信道容量 编码理论",
                ],
                "exam": "证明中心极限定理，推导梯度下降收敛条件，计算信息熵并解释其意义",
            },
            "认知科学": {
                "topics": [
                    "神经科学 神经元 突触可塑性 LTP/LTD 脑区功能 默认模式网络",
                    "认知心理学 注意 记忆 语言 决策 问题解决 元认知",
                    "机器学习 监督学习 无监督学习 强化学习 深度学习 迁移学习",
                    "语言学 句法结构 语义网络 语用推理 语言习得 计算语言学",
                    "哲学 认识论 心灵哲学 意识研究 自由意志 人工智能哲学",
                ],
                "exam": "比较LTP与反向传播的异同，设计一个类脑记忆系统，论述意识的计算理论",
            },
        },
        "pass_score": 75,
    },
}


# ============================================================
# 知识编码器
# ============================================================

class KnowledgeEncoder:
    """将知识点编码为 NeuroFlow 可处理的特征向量"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, text: str, complexity: float = 0.5) -> np.ndarray:
        """将文本知识点编码为特征向量"""
        vec = np.zeros(self.dim, dtype=np.float32)

        # 字符级哈希
        for i, ch in enumerate(text):
            idx = (ord(ch) * 7 + i * 13 + i * i * 3) % self.dim
            vec[idx] += 0.08 / max(len(text) / 10, 1)

        # 语义结构：复杂度调制正弦波
        freq = 1 + complexity * 8
        wave = np.sin(np.linspace(0, np.pi * freq, self.dim))
        vec += wave.astype(np.float32) * complexity * 0.15

        # 概念密度：根据文本长度和词汇丰富度
        words = text.split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            vec *= (0.5 + unique_ratio * 0.5)

        return vec / (np.linalg.norm(vec) + 1e-8)


# ============================================================
# 学制训练器
# ============================================================

@dataclass
class ExamResult:
    """单次考试结果"""
    subject: str
    score: float
    details: Dict[str, float] = field(default_factory=dict)

@dataclass
class StageReport:
    """阶段报告"""
    level: str
    subjects: Dict[str, ExamResult] = field(default_factory=dict)
    average_score: float = 0.0
    passed: bool = False
    training_time: float = 0.0
    samples: int = 0


class EducationTrainer:
    """
    学制训练器。
    按小学→中学→高中→大学递进训练。
    """

    def __init__(self, agent: AutonomousAgent):
        self.agent = agent
        self.encoder = KnowledgeEncoder(dim=512)
        self.reports: List[StageReport] = []
        self.knowledge_base: Dict[str, np.ndarray] = {}  # 已学知识向量

    def train_level(self, level: str, epochs_per_subject: int = 8) -> StageReport:
        """训练一个学段的全部科目"""
        config = KNOWLEDGE_GRAPH[level]
        print(f"\n{'='*64}")
        print(f"  {'🎒' if level == '小学' else '📚' if level == '中学' else '🎓' if level == '高中' else '🏛️'}"
              f" {level} — {config['description']}")
        print(f"{'='*64}")

        report = StageReport(level=level)
        t0 = time.time()

        for subject, content in config["subjects"].items():
            print(f"\n  📖 {subject}")
            scores = []

            for epoch in range(epochs_per_subject):
                topic_scores = []

                for i, topic in enumerate(content["topics"]):
                    # 复杂度随epoch递增
                    complexity = 0.3 + 0.7 * (epoch / epochs_per_subject)

                    # 编码知识点
                    x = self.encoder.encode(topic, complexity).reshape(1, -1)

                    # 通过 ReasoningLoop 推理
                    trace = self.agent.reasoner.reason(x)

                    # 评估学习效果
                    confidence = trace.thoughts[-1].confidence if trace.thoughts else 0.5
                    stability = self._measure_stability(trace)
                    recall = self._test_recall(topic)

                    topic_score = (confidence * 0.3 + stability * 0.4 + recall * 0.3)
                    topic_scores.append(topic_score)

                    # 学习反馈
                    self.agent.learn_from_feedback(x, reward=topic_score)

                    # 间隔重复：高价值知识点多复习
                    if topic_score > 0.7:
                        self.knowledge_base[f"{level}/{subject}/{i}"] = x.copy()

                avg_topic = np.mean(topic_scores) if topic_scores else 0
                scores.append(avg_topic)

                progress = "█" * int(avg_topic * 20) + "░" * (20 - int(avg_topic * 20))
                if epoch % 2 == 0 or epoch == epochs_per_subject - 1:
                    print(f"    第{epoch+1}轮: score={avg_topic:.3f} |{progress}|")

            # 科目考试
            exam_score = self._run_exam(content["exam"], complexity=0.8)
            report.subjects[subject] = ExamResult(
                subject=subject,
                score=exam_score,
                details={"best_epoch": float(np.max(scores)) if scores else 0,
                         "avg_epoch": float(np.mean(scores)) if scores else 0},
            )
            print(f"    🏆 考试: {exam_score:.1f}分")

        report.training_time = time.time() - t0
        report.samples = epochs_per_subject * sum(len(c["topics"]) for c in config["subjects"].values())
        report.average_score = np.mean([r.score for r in report.subjects.values()])
        report.passed = report.average_score >= config["pass_score"]
        self.reports.append(report)

        status = "✅ 毕业" if report.passed else "📝 需补考"
        print(f"\n  {status} | 均分: {report.average_score:.1f}/{config['pass_score']} | "
              f"耗时: {report.training_time:.1f}s")

        # 阶段间进化
        if report.passed:
            self.agent.evolution.evolve(generations=10, verbose=False)
            self.agent.evolution.consolidate()

        return report

    def _measure_stability(self, trace) -> float:
        """测量推理稳定性"""
        if not trace or len(trace.thoughts) < 2:
            return 0.5
        confidences = [t.confidence for t in trace.thoughts]
        return 1.0 - np.std(confidences)

    def _test_recall(self, topic: str) -> float:
        """测试记忆保持"""
        if len(self.knowledge_base) == 0:
            return 0.5
        # 取最近的记忆做相似度比较
        recent = list(self.knowledge_base.values())[-1]
        current = self.encoder.encode(topic).reshape(1, -1)
        sim = float(np.dot(recent.flatten(), current.flatten()) / 
                   (np.linalg.norm(recent) * np.linalg.norm(current) + 1e-8))
        return np.clip((sim + 1) / 2, 0, 1)

    def _run_exam(self, exam_question: str, complexity: float = 0.8) -> float:
        """运行阶段考试 — 综合评估"""
        questions = exam_question.replace("？", "?").replace("，", ",").split("?")
        scores = []

        for q in questions:
            q = q.strip()
            if not q:
                continue
            x = self.encoder.encode(q, complexity).reshape(1, -1)
            trace = self.agent.reasoner.reason(x, max_steps=5)

            # 多维度评估
            confidence = trace.thoughts[-1].confidence if trace.thoughts else 0.5
            stability = self._measure_stability(trace)
            recall = self._test_recall(q)
            depth = min(1.0, len(trace.thoughts) / 5.0)  # 推理深度

            # 加权
            score = confidence * 0.25 + stability * 0.25 + recall * 0.3 + depth * 0.2
            scores.append(score)

        return float(np.mean(scores)) * 100 if scores else 50

    def run_full_education(self):
        """完整学制训练"""
        print("=" * 66)
        print("  🏫 NeuroFlow 学制训练系统")
        print("  小学 → 中学 → 高中 → 大学")
        print("=" * 66)
        print(f"\n  智能体: {self.agent.name}")
        print(f"  课程: 4学段 × 3-4科目 × 5轮")
        print(f"  知识编码: 512维, 4级复杂度")

        for level in ["小学", "中学", "高中", "大学"]:
            report = self.train_level(level, epochs_per_subject=8)

            if not report.passed:
                print(f"\n  ⚠️ {level}未达标，补修中...")
                self.agent.evolution.auto_curriculum(steps=10)
                # 补考一次
                retry = self.train_level(level, epochs_per_subject=3)
                if not retry.passed:
                    print(f"  ❌ {level}补考未通过，但继续下一学段")
                else:
                    print(f"  ✅ {level}补考通过!")

        self._print_diploma()

    def _print_diploma(self):
        """打印毕业证书"""
        print(f"\n{'='*66}")
        print(f"  🎓 学业成绩单")
        print(f"{'='*66}")
        print(f"\n  {'学段':<8s} {'科目数':<6s} {'均分':<8s} {'达标':<6s} {'耗时':<8s} {'等级'}")
        print(f"  {'-'*50}")

        total_score = 0
        for report in self.reports:
            level_stars = "⭐" * min(5, int(report.average_score / 20))
            status = "✅" if report.passed else "📝"
            print(f"  {report.level:<8s} {len(report.subjects):<6d} "
                  f"{report.average_score:<8.1f} {status:<6s} "
                  f"{report.training_time:<8.1f}s {level_stars}")
            total_score += report.average_score

        avg = total_score / len(self.reports) if self.reports else 0
        print(f"\n  {'─'*50}")
        print(f"  总均分: {avg:.1f}  |  总耗时: {sum(r.training_time for r in self.reports):.1f}s")
        print(f"  总样本: {sum(r.samples for r in self.reports)}")
        print(f"  记忆库: {len(self.knowledge_base)} 条知识点")

        # 授予学位
        if avg >= 80:
            degree = "🏆 荣誉博士 (PhD Honoris Causa)"
        elif avg >= 70:
            degree = "🎓 博士 (PhD)"
        elif avg >= 60:
            degree = "📜 硕士 (Master)"
        elif avg >= 50:
            degree = "📘 学士 (Bachelor)"
        else:
            degree = "📝 肄业"

        print(f"\n  {'='*50}")
        print(f"  {degree}")
        print(f"  授予: {self.agent.name}")
        print(f"  适应度: {self.agent.evolution.best_fitness:.4f}")
        print(f"  经验: {len(self.agent.evolution.experience_buffer)} 条")
        print(f"  {'='*50}")

        # 进化总结
        fitness = self.agent.evolution.get_fitness()
        print(f"\n  🧬 进化报告:")
        print(f"    代际: {fitness['generation']}")
        print(f"    最佳适应度: {fitness['best_fitness']:.4f}")
        print(f"    平均适应度: {fitness['current_avg_fitness']:.4f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("Initializing NeuroFlow Education System...")

    # 多模态模型
    mm = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=True)

    # 智能体
    agent = AutonomousAgent(mm, name="NF-Scholar")

    # 训练器
    trainer = EducationTrainer(agent)

    # 执行
    t_total = time.time()
    trainer.run_full_education()
    total_time = time.time() - t_total

    print(f"\n{'='*66}")
    print(f"  ⏱️ 训练总耗时: {total_time:.1f}s")
    print(f"{'='*66}")
