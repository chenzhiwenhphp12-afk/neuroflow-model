"""
NeuroFlow Gym 基准测试
========================
在 OpenAI Gym 标准环境中评估 NeuroSymbolicReasoner 的决策能力。
与随机策略对比，输出 Reward 曲线。

环境: CartPole-v1, LunarLander-v3 (可选)
"""
import sys, os, time, json, random, numpy as np
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import create_multimodal
from neuroflow.cognition import NeuroSymbolicReasoner, RuleEngine, RuleContext, RuleResult


# ============================================================
# Gym 规则引擎：将 Gym 观察空间映射为 Value 驱动的决策
# ============================================================
class GymRuleEngine(RuleEngine):
    """
    通用 Gym 规则引擎 — 修正版。
    
    规则:
    1. Epsilon-greedy: ε 概率随机探索，(1-ε) 概率选最优动作
    2. decision tie-break: 多个动作 confidence 相同时随机选
    3. 自适应: 统计最近 reward 趋势，reward 下降时提高 epsilon
    
    修正 (v2):
    - 移除双重 epsilon（expand + select_best 各检查一次 → 仅 select_best）
    - 修复 tie-break: 相等时随机选（之前 max 返回第一个 → 总选 action 0）
    - 动态 epsilon: 无进展时自动提高探索率
    """
    def __init__(self, n_actions: int = 2, epsilon: float = 0.15):
        super().__init__("GymRules")
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.base_epsilon = epsilon
        self._last_rewards = []
        self._strategy = "exploit"  # exploit | explore
    
    def expand(self, ctx: RuleContext) -> List[RuleResult]:
        decision = ctx.decision.flatten()[:self.n_actions]
        value_scalar = float(np.mean(ctx.value))
        
        candidates = []
        
        for idx in range(self.n_actions):
            action = np.zeros((1, max(self.n_actions, 10)), dtype=np.float32)
            action[0, idx] = 1.0
            # advantage: 相对于最大值的偏移（全相等时为 all 0 → tie）
            advantage = float(decision[idx]) - float(np.max(decision)) if self.n_actions > 1 else 1.0
            confidence = np.clip(0.3 + advantage * 2 + value_scalar * 0.2, 0.1, 0.95)
            
            candidates.append(RuleResult(
                action=action,
                confidence=float(confidence),
                explanation=f"Gym-action={idx}: dec={float(decision[idx]):.4f} adv={advantage:.3f}"
            ))
        
        return candidates
    
    def select_best(self, ctx: RuleContext, candidates: List[RuleResult]) -> RuleResult:
        # 单次 epsilon-greedy 决策
        if random.random() < self.epsilon:
            # 探索：随机选一个动作
            return random.choice(candidates)
        
        # 利用：选最高 confidence
        # 修复 tie-break：多个最高 confidence 时随机选
        max_conf = max(c.confidence for c in candidates)
        best_candidates = [c for c in candidates if c.confidence >= max_conf - 1e-8]
        return random.choice(best_candidates)
    
    def update_epsilon(self, recent_rewards: list):
        """根据最近表现动态调整 epsilon"""
        if len(recent_rewards) < 5:
            return
        self._last_rewards = recent_rewards[-10:]
        avg = float(np.mean(self._last_rewards))
        # 如果最近表现差（低于随机基线 ~22），提高探索率
        if avg < 15:
            self.epsilon = min(0.5, self.base_epsilon * 3)
        elif avg < 25:
            self.epsilon = min(0.4, self.base_epsilon * 2)
        else:
            self.epsilon = self.base_epsilon


# ============================================================
# 基准测试运行器
# ============================================================
@dataclass
class BenchmarkResult:
    env_name: str
    episodes: int
    neuroflow_rewards: List[float] = field(default_factory=list)
    random_rewards: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    
    @property
    def nf_avg(self): return float(np.mean(self.neuroflow_rewards)) if self.neuroflow_rewards else 0
    @property
    def nf_std(self): return float(np.std(self.neuroflow_rewards)) if self.neuroflow_rewards else 0
    @property
    def rand_avg(self): return float(np.mean(self.random_rewards)) if self.random_rewards else 0
    @property
    def nf_best(self): return float(np.max(self.neuroflow_rewards)) if self.neuroflow_rewards else 0
    @property
    def improvement(self): 
        if self.rand_avg == 0: return 0
        return (self.nf_avg - self.rand_avg) / abs(self.rand_avg) * 100


def obs_to_input(observation, input_dim=512):
    """将 Gym 观察向量映射到 NeuroFlow 输入空间"""
    obs = np.array(observation, dtype=np.float32)
    obs_norm = obs / (np.abs(obs).max() + 1e-8)
    # 重复填充到 512 维
    repeats = input_dim // len(obs_norm) + 1
    x = np.tile(obs_norm, repeats)[:input_dim].astype(np.float32)
    return x.reshape(1, -1)


def run_benchmark(env_name: str = "CartPole-v1", episodes: int = 50) -> BenchmarkResult:
    """
    在指定 Gym 环境中运行基准测试。
    
    Returns:
        BenchmarkResult 包含 NeuroFlow vs Random 的 Reward 对比
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("⚠️ gymnasium 未安装，尝试安装...")
        os.system(f"{sys.executable} -m pip install gymnasium -q")
        import gymnasium as gym
    
    result = BenchmarkResult(env_name=env_name, episodes=episodes)
    t0 = time.time()
    
    print(f"🏋️  {env_name} — {episodes} episodes")
    print(f"{'='*55}")
    
    # 初始化模型和推理器
    model = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=True)
    
    env = gym.make(env_name)
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    engine = GymRuleEngine(n_actions=n_actions, epsilon=0.15)
    reasoner = NeuroSymbolicReasoner(model, engine)
    
    print(f"  Actions: {n_actions}  |  Obs dim: {obs_dim}")
    
    # === NeuroFlow 运行 ===
    print(f"\n  [NeuroFlow]")
    nf_total_steps = 0
    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0
        ep_actions = []
        
        while not done and ep_steps < 500:
            x = obs_to_input(obs)
            trace = reasoner.reason(x, max_steps=3, confidence_threshold=0.5)
            
            # 提取动作
            if trace.final_action is not None:
                action = int(np.argmax(trace.final_action.flatten()[:n_actions]))
            else:
                action = env.action_space.sample()
            
            ep_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
        
        result.neuroflow_rewards.append(ep_reward)
        nf_total_steps += ep_steps
        
        # 自适应 epsilon（根据最近表现调整探索率）
        engine.update_epsilon(result.neuroflow_rewards)
        
        if (ep + 1) % 10 == 0:
            avg = np.mean(result.neuroflow_rewards[-10:])
            zeros = ep_actions.count(0)
            ones = ep_actions.count(1)
            print(f"    Ep {ep+1:3d}/{episodes} | avg={avg:.1f} | best={result.nf_best:.0f} | "
                  f"ε={engine.epsilon:.2f} | actions L/R={zeros}/{ones}")
    
    # === Random 基线 ===
    print(f"\n  [Random baseline]")
    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        ep_steps = 0
        
        while not done and ep_steps < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
        
        result.random_rewards.append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            avg = np.mean(result.random_rewards[-10:])
            print(f"    Ep {ep+1:3d}/{episodes} | avg_reward={avg:.1f}")
    
    env.close()
    result.elapsed_seconds = time.time() - t0
    
    # 打印结果
    print(f"\n{'='*55}")
    print(f"📊 基准测试结果: {env_name}")
    print(f"{'='*55}")
    print(f"  NeuroFlow:  avg={result.nf_avg:8.1f} ±{result.nf_std:5.1f}  best={result.nf_best:.0f}")
    print(f"  Random:     avg={result.rand_avg:8.1f} ±{np.std(result.random_rewards):5.1f}")
    print(f"  提升:       {result.improvement:+.1f}%")
    print(f"  耗时:       {result.elapsed_seconds:.1f}s")
    
    # 保存结果
    out = {
        "env": env_name,
        "episodes": episodes,
        "neuroflow": {"avg": result.nf_avg, "std": result.nf_std, "best": result.nf_best},
        "random": {"avg": result.rand_avg, "std": float(np.std(result.random_rewards))},
        "improvement_pct": result.improvement,
        "elapsed_s": result.elapsed_seconds,
    }
    outpath = f"/mnt/d/neuroflow-model/benchmarks/{env_name.lower()}_result.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  💾 Saved: {outpath}")
    
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    
    run_benchmark(args.env, args.episodes)
