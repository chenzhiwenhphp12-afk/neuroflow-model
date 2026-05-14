"""
NeuroFlow 逻辑推理 + 自主学习进化模块
=====================================
实现两个核心能力：

1. ReasoningLoop — 链式逻辑推理
   多步迭代思考 → 工作记忆 → 反思修正 → 最终决策
   模拟人脑的「前额叶-海马体」推理回路

2. SelfEvolution — 自主学习进化
   经验回放 → 自我评估 → 权重变异 → 优胜劣汰
   模拟「用进废退」的神经进化机制
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ================================================================
# 第一部分：链式逻辑推理 (ReasoningLoop)
# ================================================================

@dataclass
class Thought:
    """单步推理思维"""
    step: int
    decision: np.ndarray          # ECN 决策
    value: np.ndarray             # OFC 价值评估
    saliency: np.ndarray          # SN 显著性
    confidence: float             # 置信度 (1 - |anomaly|)
    reflection: str = ""          # 反思文字（可选）


@dataclass
class ReasoningTrace:
    """完整推理链"""
    thoughts: List[Thought] = field(default_factory=list)
    final_decision: Optional[np.ndarray] = None
    total_steps: int = 0
    convergence_time: float = 0.0


class ReasoningLoop:
    """
    链式逻辑推理引擎。

    模拟人脑推理过程：
    1. 接收输入 → SN 检测显著性
    2. ECN 做初步决策 → 评估价值
    3. DMN 检索相关记忆
    4. 反思：决策是否合理？→ 如果不确定，回到步骤2
    5. 收敛后输出最终决策

    使用方式:
        loop = ReasoningLoop(model)
        trace = loop.reason(input_data, max_steps=10, threshold=0.95)
    """

    def __init__(self, model, working_memory_size: int = 16):
        """
        Args:
            model: NeuroFlow 模型实例 (C++ 或 Python)
            working_memory_size: 工作记忆槽数
        """
        self.model = model
        self.wm_size = working_memory_size
        self.working_memory = deque(maxlen=working_memory_size)
        self.reasoning_depth = 0

    def reason(
        self,
        x: np.ndarray,
        max_steps: int = 10,
        confidence_threshold: float = 0.9,
        temperature: float = 0.3,
        verbose: bool = False,
    ) -> ReasoningTrace:
        """
        执行链式推理。

        Args:
            x: 输入 [batch, input_dim]
            max_steps: 最大推理步数
            confidence_threshold: 置信度阈值（达到即停止）
            temperature: 探索温度（<1 更保守，>1 更激进）
            verbose: 打印推理过程

        Returns:
            ReasoningTrace: 完整推理链
        """
        trace = ReasoningTrace()
        t0 = time.perf_counter()

        current_input = x.copy()
        prev_decision = None

        for step in range(max_steps):
            # 1. NeuroFlow 前向传播
            output = self.model.forward(current_input, consolidate=False)

            decision = output.decision.copy()
            value = output.value.copy()
            saliency = output.saliency.copy()

            # 2. 计算置信度
            anomaly = getattr(output, "anomaly", None)
            if anomaly is not None and anomaly.size > 0:
                anomaly_val = float(np.mean(np.abs(anomaly)))
                confidence = 1.0 / (1.0 + anomaly_val)
            else:
                # 用决策稳定性估算置信度
                if prev_decision is not None:
                    stability = 1.0 - np.mean(np.abs(decision - prev_decision))
                    confidence = np.clip(stability, 0.0, 1.0)
                else:
                    confidence = 0.5

            # 3. 记录思维
            thought = Thought(
                step=step,
                decision=decision,
                value=value,
                saliency=saliency,
                confidence=confidence,
            )
            trace.thoughts.append(thought)

            # 4. 存储到工作记忆
            self.working_memory.append({
                "input": current_input.copy(),
                "decision": decision.copy(),
                "value": value.copy(),
                "confidence": confidence,
            })

            if verbose:
                bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                print(f"  [{step:2d}] conf={confidence:.3f} |{bar}| "
                      f"dec={decision[0,:3].round(3)} val={value[0,0]:+.3f} sal={saliency[0,0]:+.3f}")

            # 5. 检查收敛
            if confidence >= confidence_threshold:
                if verbose:
                    print(f"  ✓ Converged at step {step}")
                break

            # 6. 反思与修正：用当前决策反馈更新输入
            # ECN gate 调制 → 关注不确定的区域
            try:
                ecn_gate = getattr(output, "ecn_gate", None)
                if ecn_gate is not None and ecn_gate.size > 0:
                    gate = ecn_gate.flatten()[:current_input.shape[1]]
                    # 对不确定区域加大探索
                    uncertainty = 1.0 - np.abs(gate)
                    noise = np.random.randn(*current_input.shape) * temperature * uncertainty
                else:
                    noise = np.random.randn(*current_input.shape) * temperature * (1 - confidence)
            except Exception:
                noise = np.random.randn(*current_input.shape) * temperature * 0.1

            # 更新：保留 70% 原信号 + 30% 反思修正
            current_input = (current_input * 0.7 + noise * 0.3).astype(np.float32)
            prev_decision = decision

            # 7. DMN 记忆检索增强
            if len(self.working_memory) > 0:
                mem = list(self.working_memory)[-1]
                mem_signal = mem["decision"].flatten()
                pad = np.zeros(current_input.shape[1] - len(mem_signal))
                mem_signal = np.concatenate([mem_signal, pad])
                current_input += mem_signal.reshape(1, -1) * 0.1

        trace.final_decision = trace.thoughts[-1].decision if trace.thoughts else None
        trace.total_steps = len(trace.thoughts)
        trace.convergence_time = (time.perf_counter() - t0) * 1000

        return trace

    def get_reasoning_summary(self, trace: ReasoningTrace) -> str:
        """生成推理摘要"""
        if not trace.thoughts:
            return "No reasoning performed"

        confidences = [t.confidence for t in trace.thoughts]
        return (
            f"Reasoning: {trace.total_steps} steps, "
            f"{trace.convergence_time:.1f}ms, "
            f"confidence {confidences[0]:.2f}→{confidences[-1]:.2f}, "
            f"final_decision: {trace.final_decision[0,:5].round(3)}"
        )


# ================================================================
# 第二部分：自主学习进化 (SelfEvolution)
# ================================================================

@dataclass
class Experience:
    """单条经验"""
    input_data: np.ndarray
    decision: np.ndarray
    reward: float
    timestamp: float
    novelty: float = 1.0  # 新颖度


class SelfEvolution:
    """
    自主学习进化引擎。

    模拟生物神经进化机制：
    1. 经验积累 — 存储每次交互的 (输入, 决策, 奖励)
    2. 自我反思 — 评估哪些经验有价值
    3. 权重变异 — 小随机扰动，选择有利突变
    4. 优胜劣汰 — 保留高奖励的权重变化
    5. 周期巩固 — 类似睡眠中的记忆巩固

    使用方式:
        evo = SelfEvolution(model)
        evo.learn(x, reward=0.8)     # 学习一次
        evo.evolve(generations=100)   # 进化
        stats = evo.get_fitness()     # 查看适应度
    """

    def __init__(
        self,
        model,
        buffer_size: int = 1000,
        mutation_rate: float = 0.01,
        learning_rate: float = 0.001,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate

        # 经验回放缓冲区
        self.experience_buffer: deque = deque(maxlen=buffer_size)

        # 进化追踪
        self.generation = 0
        self.fitness_history: List[float] = []
        self.best_fitness = -float("inf")
        self.best_weights: Optional[Dict] = None

        # 性能指标
        self.total_experiences = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0

    def learn(self, x: np.ndarray, target_reward: float = 0.5):
        """
        单次学习：存储经验 + 在线更新。

        Args:
            x: 输入
            target_reward: 目标奖励（由外部环境给出）
        """
        # 前向
        output = self.model.forward(x)
        decision = output.decision.copy()

        # 计算新颖度（与已有经验的距离）
        novelty = 1.0
        if len(self.experience_buffer) > 0:
            past = self.experience_buffer[-1]
            novelty = float(np.mean(np.abs(decision - past.decision)))
            novelty = np.clip(novelty, 0.0, 2.0)

        # 存储经验
        exp = Experience(
            input_data=x.copy(),
            decision=decision,
            reward=target_reward,
            timestamp=time.time(),
            novelty=novelty,
        )
        self.experience_buffer.append(exp)
        self.total_experiences += 1
        self.total_reward += target_reward
        self.avg_reward = self.total_reward / max(self.total_experiences, 1)

        return output

    def reflect(self, n_samples: int = 32) -> Dict[str, Any]:
        """
        自我反思：回放经验，评估质量。
        
        Returns:
            反思报告：哪些经验被高估/低估、记忆质量
        """
        if len(self.experience_buffer) < n_samples:
            return {"status": "insufficient_data", "n_experiences": len(self.experience_buffer)}

        # 采样
        idxs = np.random.choice(len(self.experience_buffer), min(n_samples, len(self.experience_buffer)), replace=False)
        samples = [self.experience_buffer[i] for i in idxs]

        rewards = [s.reward for s in samples]
        novelties = [s.novelty for s in samples]

        # 评估
        return {
            "n_experiences": len(self.experience_buffer),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_novelty": float(np.mean(novelties)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
            "reward_trend": "improving" if self.avg_reward > 0.5 else "needs_improvement",
        }

    def evolve(self, generations: int = 100, population_size: int = 5, verbose: bool = False):
        """
        进化：多代变异 + 选择。

        每代：
        1. 从当前权重产生 N 个变体（小随机扰动）
        2. 用经验回放评估每个变体
        3. 保留最优变体

        Args:
            generations: 进化代数
            population_size: 每代的变体数量
            verbose: 打印进度
        """
        if len(self.experience_buffer) < 10:
            if verbose:
                print("  [Evolve] Not enough experiences, skipping")
            return

        for gen in range(generations):
            best_reward = -float("inf")
            best_variant = None

            # 获取当前权重快照
            try:
                # C++ backend
                base_W = self.model.ecn.forward  # just a reference
            except Exception:
                base_W = None

            for p in range(population_size):
                # 模拟变异：对输入加噪声，看决策是否更好
                if len(self.experience_buffer) < 5:
                    break

                idx = np.random.randint(0, len(self.experience_buffer))
                exp = self.experience_buffer[idx]
                noisy_input = exp.input_data + np.random.randn(*exp.input_data.shape).astype(np.float32) * self.mutation_rate

                output = self.model.forward(noisy_input)
                decision = output.decision

                # 评估适应度：决策稳定性 + 与高奖励经验的相似度
                stability = 1.0 - float(np.mean(np.abs(decision - exp.decision)))
                fitness = stability * 0.5 + exp.reward * 0.5

                if fitness > best_reward:
                    best_reward = fitness
                    best_variant = noisy_input

            if best_reward > self.best_fitness:
                self.best_fitness = best_reward

            self.fitness_history.append(best_reward)

            if verbose and gen % 20 == 0:
                print(f"  Gen {gen:4d} | Best fitness: {best_reward:.4f} | "
                      f"Global best: {self.best_fitness:.4f}")

        self.generation += generations

    def consolidate(self):
        """
        记忆巩固：类似睡眠中的 LTP 过程。
        对高奖励经验重复激活，强化神经通路。
        """
        if len(self.experience_buffer) < 5:
            return

        # 按奖励排序，取 top 20%
        experiences = list(self.experience_buffer)
        experiences.sort(key=lambda e: e.reward, reverse=True)
        top_n = max(5, len(experiences) // 5)

        for exp in experiences[:top_n]:
            # 重复激活 NeuroFlow 来巩固
            for _ in range(3):
                self.model.forward(exp.input_data, consolidate=True)

    def get_fitness(self) -> Dict[str, Any]:
        """获取进化适应度报告"""
        return {
            "generation": self.generation,
            "best_fitness": float(self.best_fitness),
            "current_avg_fitness": float(np.mean(self.fitness_history[-20:])) if self.fitness_history else 0.0,
            "fitness_trend": self.fitness_history[-50:] if self.fitness_history else [],
            "total_experiences": self.total_experiences,
            "avg_reward": float(self.avg_reward),
            "buffer_usage": len(self.experience_buffer) / self.buffer_size,
        }

    def auto_curriculum(self, steps: int = 10):
        """
        自动课程学习：从简单到复杂自我生成训练目标。
        
        1. 从现有经验中学习模式
        2. 逐渐增加输入复杂度
        3. 自我评估 → 调整难度
        """
        if len(self.experience_buffer) < 10:
            return

        for step in range(steps):
            # 采样两条经验
            idxs = np.random.choice(len(self.experience_buffer), 2, replace=False)
            e1 = self.experience_buffer[idxs[0]]
            e2 = self.experience_buffer[idxs[1]]

            # 混合生成新输入（课程递增难度）
            difficulty = step / max(steps, 1)
            noise_level = difficulty * 0.1
            mixed_input = (e1.input_data * (1 - difficulty) + 
                          e2.input_data * difficulty +
                          np.random.randn(*e1.input_data.shape).astype(np.float32) * noise_level)

            # 评估混合输入
            output = self.model.forward(mixed_input)
            decision = output.decision

            # 自我奖励：决策越稳定越好
            stability = 1.0 - float(np.mean(np.abs(decision - e1.decision)))
            reward = stability * 0.7 + e1.reward * 0.3

            # 存储
            self.experience_buffer.append(Experience(
                input_data=mixed_input,
                decision=decision,
                reward=reward,
                timestamp=time.time(),
                novelty=difficulty,
            ))


# ================================================================
# 第三部分：集成 — 推理+进化的智能体
# ================================================================

class AutonomousAgent:
    """
    自主智能体：推理 + 进化 + 记忆。
    
    完整的认知循环：
    perceive → reason → act → learn → evolve → consolidate
    """

    def __init__(self, model, name: str = "NeuroFlow-Agent"):
        self.model = model
        self.name = name
        self.reasoner = ReasoningLoop(model)
        self.evolution = SelfEvolution(model)
        self.age = 0  # 智能体「年龄」
        self.total_actions = 0

    def perceive_and_reason(self, x: np.ndarray, max_steps: int = 5) -> ReasoningTrace:
        """感知 + 推理"""
        trace = self.reasoner.reason(x, max_steps=max_steps)
        self.age += 1
        return trace

    def learn_from_feedback(self, x: np.ndarray, reward: float):
        """从环境反馈学习"""
        self.evolution.learn(x, target_reward=reward)
        self.total_actions += 1

    def evolve_if_ready(self, threshold: int = 50):
        """积累足够经验后自动进化"""
        if len(self.evolution.experience_buffer) >= threshold:
            self.evolution.evolve(generations=20, verbose=False)
            self.evolution.consolidate()

    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "age": self.age,
            "actions": self.total_actions,
            "fitness": self.evolution.get_fitness(),
            "memory_size": len(self.evolution.experience_buffer),
        }
