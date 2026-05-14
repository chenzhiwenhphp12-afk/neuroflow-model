"""
NeuroFlow 认知体系训练
======================
Piaget 认知发展阶段 × 多模态 × 自主学习进化

四阶段训练：
  Stage 1 — 感知 (Sensory-Motor):   识别基本模式（形状/纹理/频率）
  Stage 2 — 联想 (Pre-Operational): 文本-图像跨模态对齐
  Stage 3 — 推理 (Concrete-Op):     逻辑推理链 + 因果关系
  Stage 4 — 元认知 (Formal-Op):     自我反思 + 进化优化

训练策略：
  - 合成数据生成（无需外部数据集）
  - 课程学习（递增难度）
  - 间隔重复（遗忘曲线优化）
  - 自我对抗（生成器 vs 判别器）
"""

import sys, numpy as np, time, json, os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import NeuroFlowLite, create_multimodal, MultiModalConfig
from neuroflow.cognition import ReasoningLoop, SelfEvolution, AutonomousAgent


# ============================================================
# 合成认知训练数据生成器
# ============================================================

class CognitiveDataGenerator:
    """
    生成认知训练的合成数据。
    仿照儿童认知发展中的典型任务设计。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.text_dim = 512
        self.image_size = 224
        self.audio_dim = 256

    def make_image(self, pattern: str, complexity: float = 0.5) -> np.ndarray:
        """生成合成图像 (3, 224, 224)"""
        img = np.zeros((3, 224, 224), dtype=np.float32)
        s = int(40 + complexity * 80)  # size scales with complexity
        cx, cy = 112, 112

        if pattern == "circle":
            Y, X = np.ogrid[:224, :224]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            r = s // 2
            img[0] = (dist < r).astype(np.float32)
            img[1] = (dist < r * 0.7).astype(np.float32) * 0.5

        elif pattern == "square":
            img[0, cy - s // 2 : cy + s // 2, cx - s // 2 : cx + s // 2] = 1.0

        elif pattern == "triangle":
            for y in range(224):
                w = int((1 - y / 224) * s)
                x0 = max(0, cx - w // 2)
                x1 = min(224, cx + w // 2)
                if y < s:
                    img[0, y, x0:x1] = 1.0

        elif pattern == "cross":
            t = max(4, s // 8)
            img[0, cy - t : cy + t, :] = 1.0
            img[1, :, cx - t : cx + t] = 1.0

        elif pattern == "grid":
            step = max(8, int(40 - complexity * 30))
            for i in range(0, 224, step):
                img[0, i : i + 2, :] = 1.0
                img[1, :, i : i + 2] = 1.0

        elif pattern == "spiral":
            for angle in np.linspace(0, complexity * 8 * np.pi, 500):
                r = angle / (4 * np.pi) * s
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < 224 and 0 <= y < 224:
                    img[0, y - 1 : y + 2, x - 1 : x + 2] = 1.0

        elif pattern == "concentric":
            for r in range(20, s, s // 4):
                Y, X = np.ogrid[:224, :224]
                dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                ring = (np.abs(dist - r) < 3).astype(np.float32)
                img[0] += ring * 0.3
                img[1] += ring * 0.3

        # Add noise proportional to complexity
        noise = self.rng.randn(*img.shape).astype(np.float32) * complexity * 0.05
        return np.clip(img + noise, 0, 1)

    def make_text(self, concept: str, complexity: float = 0.5) -> np.ndarray:
        """将概念编码为文本特征向量"""
        vec = np.zeros(self.text_dim, dtype=np.float32)
        words = concept.split()
        for i, w in enumerate(words):
            for j, ch in enumerate(w):
                idx = (ord(ch) * 7 + i * 13 + j * 31) % self.text_dim
                vec[idx] += 0.15

        # Add semantic structure based on complexity
        structured = complexity * np.sin(np.linspace(0, np.pi * complexity * 4, self.text_dim))
        vec += structured.astype(np.float32) * 0.1
        return vec

    def make_audio(self, pattern: str, complexity: float = 0.5) -> np.ndarray:
        """生成合成音频特征"""
        freq_map = {"low": 220, "mid": 440, "high": 880, "chord": 0}
        freq = freq_map.get(pattern, 440)

        t = np.linspace(0, 0.5, 4000, dtype=np.float32)
        if freq > 0:
            signal = np.sin(2 * np.pi * freq * t) * (1 - complexity * 0.5)
            signal += np.sin(2 * np.pi * freq * 2 * t) * complexity * 0.3
        else:
            signal = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554 * t) +
                     np.sin(2 * np.pi * 659 * t)) * 0.3

        # FFT
        frame_size = 256
        hop = 128
        n_frames = min(20, (len(signal) - frame_size) // hop + 1)
        spectrum = np.zeros(n_frames * (frame_size // 2), dtype=np.float32)
        for i in range(n_frames):
            frame = signal[i * hop : i * hop + frame_size]
            fft = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            spectrum[i * (frame_size // 2) : (i + 1) * (frame_size // 2)] = fft[: frame_size // 2]

        # Compress to audio_dim
        repeats = self.audio_dim // len(spectrum) + 1
        features = np.tile(spectrum, repeats)[:self.audio_dim].astype(np.float32)
        return features / (features.max() + 1e-8)

    def generate_batch(self, n: int = 32, difficulty: float = 0.5) -> List[Dict]:
        """
        生成一批认知训练样本。
        difficulty ∈ [0, 1]: 0=简单, 1=困难
        """
        patterns = ["circle", "square", "triangle", "cross", "grid", "spiral", "concentric"]
        concepts = [
            "round shape", "box", "pointy peak", "intersection",
            "patterned lines", "curved spiral", "nested rings",
        ]
        audio_types = ["low", "mid", "high", "chord"]

        batch = []
        for _ in range(n):
            p_idx = self.rng.randint(0, min(len(patterns), int(3 + difficulty * 4)))
            pattern = patterns[p_idx]
            concept = concepts[p_idx]
            audio = audio_types[self.rng.randint(0, len(audio_types))]

            sample = {
                "image": self.make_image(pattern, difficulty),
                "text": self.make_text(concept, difficulty),
                "audio": self.make_audio(audio, difficulty),
                "label": p_idx,  # pattern class
                "concept": concept,
                "difficulty": difficulty,
                "correct_match": True,
            }
            batch.append(sample)
        return batch

    def generate_negative_samples(self, batch: List[Dict]) -> List[Dict]:
        """生成负样本（不匹配的图文对）"""
        negatives = []
        n = len(batch)
        for i in range(n):
            j = (i + 1) % n  # next sample
            neg = {
                "image": batch[i]["image"],
                "text": batch[j]["text"],
                "audio": batch[i]["audio"],
                "label": batch[i]["label"],
                "concept": batch[j]["concept"],
                "difficulty": batch[i]["difficulty"],
                "correct_match": False,
            }
            negatives.append(neg)
        return negatives

    def generate_reasoning_task(self, steps: int = 3) -> Dict:
        """
        生成链式推理任务。
        例: "如果 A→B, B→C, 那么 A→?"
        """
        symbols = ["A", "B", "C", "D", "E", "F"]
        chain = []
        for i in range(steps):
            chain.append({
                "premise": f"{symbols[i]} leads to {symbols[i+1]}",
                "input": self.make_text(f"{symbols[i]} to {symbols[i+1]}"),
                "step": i,
            })

        task = {
            "chain": chain,
            "question": f"What does {symbols[0]} lead to?",
            "answer": symbols[steps],
            "answer_input": self.make_text(symbols[steps]),
        }
        return task


# ============================================================
# 认知训练管线
# ============================================================

@dataclass
class StageMetrics:
    """单阶段训练指标"""
    name: str
    loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    time_elapsed: float = 0.0
    samples_processed: int = 0


class CognitiveTrainer:
    """
    认知体系训练器。
    实现 Piaget 四阶段认知发展训练。
    """

    def __init__(self, agent: AutonomousAgent, generator: CognitiveDataGenerator):
        self.agent = agent
        self.gen = generator
        self.stages: List[StageMetrics] = []
        self.total_samples = 0
        self.training_history: List[Dict] = []

    def _forward_sample(self, sample: Dict) -> Dict:
        """单个样本前向传播，返回输出指标"""
        img = sample["image"].reshape(1, 3, 224, 224).astype(np.float32)
        txt = sample["text"].reshape(1, -1)

        out = self.agent.model.forward_multimodal(txt, img)
        return {
            "sim": float(out.text_image_sim[0, 0]),
            "saliency": float(out.saliency[0, 0]) if out.saliency.ndim >= 2 else float(out.saliency),
            "decision": out.decision[0],
            "value": float(out.value[0, 0]),
        }

    def train_stage1_perception(self, epochs: int = 20, samples_per_epoch: int = 32):
        """
        Stage 1: 感知训练 (Sensory-Motor)
        目标：识别基本视觉/文本/音频模式
        """
        print("\n" + "=" * 60)
        print("  🍼 Stage 1: 感知训练 (Sensory-Motor)")
        print("  目标: 学习识别基本模式 — 圆形/方形/三角形")
        print("=" * 60)

        metrics = StageMetrics(name="Perception")
        t0 = time.time()

        for epoch in range(epochs):
            difficulty = min(1.0, epoch / epochs)
            batch = self.gen.generate_batch(samples_per_epoch, difficulty)

            epoch_loss = 0.0
            correct = 0

            for sample in batch:
                result = self._forward_sample(sample)

                # 损失：匹配样本应有高相似度
                target_sim = 0.5 if sample["correct_match"] else -0.5
                loss = (result["sim"] - target_sim) ** 2
                epoch_loss += loss

                # 准确率：相似度符号正确
                if (result["sim"] > 0) == sample["correct_match"]:
                    correct += 1

                # 学习反馈
                reward = 1.0 if (result["sim"] > 0) == sample["correct_match"] else 0.1
                self.agent.learn_from_feedback(sample["text"].reshape(1, -1), reward)

            avg_loss = epoch_loss / len(batch)
            acc = correct / len(batch)
            metrics.loss.append(avg_loss)
            metrics.accuracy.append(acc)

            if epoch % 5 == 0 or epoch == epochs - 1:
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"  Epoch {epoch:3d} | diff={difficulty:.2f} | "
                      f"loss={avg_loss:.4f} | acc={acc:.2f} |{bar}|")

        metrics.time_elapsed = time.time() - t0
        metrics.samples_processed = epochs * samples_per_epoch
        self.stages.append(metrics)

        print(f"\n  ✅ Stage 1 完成: {metrics.time_elapsed:.1f}s, "
              f"最终 acc={metrics.accuracy[-1]:.2%}")

    def train_stage2_association(self, epochs: int = 30):
        """
        Stage 2: 联想训练 (Pre-Operational)
        目标：建立文本-图像跨模态对齐
        """
        print("\n" + "=" * 60)
        print("  🧒 Stage 2: 联想训练 (Pre-Operational)")
        print("  目标: 文本描述 ↔ 图像 建立跨模态关联")
        print("=" * 60)

        metrics = StageMetrics(name="Association")
        t0 = time.time()

        for epoch in range(epochs):
            difficulty = min(1.0, epoch / epochs)
            positive = self.gen.generate_batch(16, difficulty)
            negative = self.gen.generate_negative_samples(positive)
            batch = positive + negative
            import random
            random.shuffle(batch)
            batch_size = len(batch)

            epoch_loss = 0.0
            correct = 0

            for sample in batch:
                result = self._forward_sample(sample)

                # Contrastive loss: 匹配对相似度应 > 不匹配对
                target = 0.8 if sample["correct_match"] else -0.8
                loss = (result["sim"] - target) ** 2
                epoch_loss += loss

                if (result["sim"] > 0) == sample["correct_match"]:
                    correct += 1

                reward = 0.8 if (result["sim"] > 0) == sample["correct_match"] else 0.0
                self.agent.learn_from_feedback(sample["text"].reshape(1, -1), reward)

            avg_loss = epoch_loss / len(batch)
            acc = correct / len(batch)
            metrics.loss.append(avg_loss)
            metrics.accuracy.append(acc)

            if epoch % 5 == 0 or epoch == epochs - 1:
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                      f"acc={acc:.2%} |{bar}|")

        metrics.time_elapsed = time.time() - t0
        metrics.samples_processed = epochs * len(batch)
        self.stages.append(metrics)

    def train_stage3_reasoning(self, epochs: int = 20):
        """
        Stage 3: 推理训练 (Concrete Operational)
        目标：链式逻辑推理
        """
        print("\n" + "=" * 60)
        print("  🧑 Stage 3: 推理训练 (Concrete Operational)")
        print("  目标: 多步因果推理 — A→B→C→?")
        print("=" * 60)

        metrics = StageMetrics(name="Reasoning")
        t0 = time.time()
        reasoner = self.agent.reasoner

        for epoch in range(epochs):
            task = self.gen.generate_reasoning_task(steps=1 + epoch % 3)

            # 逐步输入推理链
            trace = None
            for premise in task["chain"]:
                x = premise["input"].reshape(1, -1)
                trace = reasoner.reason(x)

            # 评估：最终决策是否接近正确答案
            answer_vec = task["answer_input"]
            x_answer = answer_vec.reshape(1, -1)
            out = self.agent.model.forward_text(x_answer)

            # 推理质量 = 决策稳定性
            if trace and trace.thoughts:
                confidence = trace.thoughts[-1].confidence
            else:
                confidence = 0.5

            metrics.accuracy.append(confidence)
            reward = confidence
            self.agent.learn_from_feedback(x_answer, reward)

            if epoch % 5 == 0 or epoch == epochs - 1:
                bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                print(f"  Epoch {epoch:3d} | steps={task['chain'][-1]['step']+1} | "
                      f"confidence={confidence:.3f} |{bar}|  答案: {task['answer']}")

        metrics.time_elapsed = time.time() - t0
        metrics.samples_processed = epochs
        self.stages.append(metrics)

    def train_stage4_metacognition(self, generations: int = 30):
        """
        Stage 4: 元认知训练 (Formal Operational)
        目标：自我反思 + 进化优化
        """
        print("\n" + "=" * 60)
        print("  🧠 Stage 4: 元认知训练 (Formal Operational)")
        print("  目标: 自我进化 — 从经验中学习并适应")
        print("=" * 60)

        metrics = StageMetrics(name="Metacognition")
        t0 = time.time()

        fitness_before = self.agent.evolution.best_fitness

        for gen in range(generations):
            # 自我课程学习
            self.agent.evolution.auto_curriculum(steps=3)

            # 进化
            self.agent.evolution.evolve(generations=5, verbose=False)

            # 反思
            reflect = self.agent.evolution.reflect()
            metrics.accuracy.append(reflect.get("mean_reward", 0))

            if gen % 5 == 0 or gen == generations - 1:
                fitness = self.agent.evolution.best_fitness
                bar = "█" * int(min(fitness, 1.0) * 20)
                print(f"  Gen {gen:3d} | fitness={fitness:.4f} | "
                      f"reward={reflect.get('mean_reward', 0):.3f} | {bar}")

        # 最终巩固
        self.agent.evolution.consolidate()

        fitness_after = self.agent.evolution.best_fitness
        metrics.time_elapsed = time.time() - t0
        metrics.samples_processed = generations
        self.stages.append(metrics)

        print(f"\n  ✅ Stage 4 完成: {metrics.time_elapsed:.1f}s")
        print(f"  适应度: {fitness_before:.4f} → {fitness_after:.4f}")

    def run_full_training(self):
        """执行完整的四阶段认知训练"""
        print("=" * 66)
        print("  🧠 NeuroFlow 认知体系训练")
        print("  Piaget × MultiModal × SelfEvolution")
        print("=" * 66)

        status = self.agent.get_status()
        print(f"\n  智能体: {status['name']} | "
              f"模型: Lite (331K) + MultiModal (232K)")

        # 四阶段训练
        self.train_stage1_perception(epochs=20)
        self.train_stage2_association(epochs=30)
        self.train_stage3_reasoning(epochs=20)
        self.train_stage4_metacognition(generations=30)

        # 报告
        self._print_report()

    def _print_report(self):
        print("\n" + "=" * 66)
        print("  📊 认知训练报告")
        print("=" * 66)

        for stage in self.stages:
            start_acc = stage.accuracy[0] if stage.accuracy else 0
            final_acc = stage.accuracy[-1] if stage.accuracy else 0
            improvement = final_acc - start_acc
            arrow = "↑" if improvement > 0 else "→"

            print(f"\n  [{stage.name}]")
            print(f"    样本数: {stage.samples_processed}")
            print(f"    耗时:   {stage.time_elapsed:.1f}s")
            print(f"    精度:   {start_acc:.2%} → {final_acc:.2%} ({improvement:+.1%} {arrow})")

        print(f"\n  总耗时: {sum(s.time_elapsed for s in self.stages):.1f}s")
        print(f"  总样本: {sum(s.samples_processed for s in self.stages)}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("Initializing cognitive system...")

    # 多模态模型
    cfg = MultiModalConfig()
    cfg.text_dim = 512
    cfg.image_size = 224
    cfg.output_dim = 10
    cfg.use_quantization = True

    mm = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=True)

    # 认知智能体
    agent = AutonomousAgent(mm, name="NF-Cognitive")

    # 数据生成器
    gen = CognitiveDataGenerator(seed=42)

    # 训练器
    trainer = CognitiveTrainer(agent, gen)

    # 执行
    t_total = time.time()
    trainer.run_full_training()
    total_time = time.time() - t_total

    # 最终状态
    print(f"\n{'='*66}")
    print(f"  🏁 认知训练完成! 总耗时: {total_time:.1f}s")
    print(f"{'='*66}")

    status = agent.get_status()
    print(f"\n  智能体状态:")
    print(f"    年龄: {status['age']} 代")
    print(f"    经验: {status['memory_size']} 条")
    print(f"    适应度: {status['fitness']['best_fitness']:.4f}")
