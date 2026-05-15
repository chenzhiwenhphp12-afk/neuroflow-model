"""
NeuroFlow Gym 基准测试
========================
在 OpenAI Gym 标准环境中评估 NeuroSymbolicReasoner 的决策能力。
与随机策略对比，输出 Reward 曲线。

环境: CartPole-v1, LunarLander-v3 (可选)
"""
import sys, os, time, json, numpy as np
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
    通用 Gym 规则引擎。
    规则:
    1. 系统 1 Value > 阈值 → 选择最高 Value 维度对应的动作
    2. 如果 Value 都低 → 随机探索 (epsilon-greedy)
    3. 连续 N 步无进步 → 切换策略
    """
    def __init__(self, n_actions: int = 2, epsilon: float = 0.1):
        super().__init__("GymRules")
        self.n_actions = n_actions
        self.epsilon = epsilon
        self._last_rewards = []
        self._strategy = "exploit"  # exploit | explore
    
    def expand(self, ctx: RuleContext) -> List[RuleResult]:
        decision = ctx.decision.flatten()[:self.n_actions]
        value_scalar = float(np.mean(ctx.value))
        
        candidates = []
        
        for idx in range(self.n_actions):
            action = np.zeros((1, max(self.n_actions, 10)), dtype=np.float32)
            action[0, idx] = 1.0
            advantage = float(decision[idx]) - float(np.max(decision)) if self.n_actions > 1 else 1.0
            confidence = np.clip(0.3 + advantage * 2 + value_scalar * 0.2, 0.1, 0.95)
            
            candidates.append(RuleResult(
                action=action,
                confidence=float(confidence),
                explanation=f"Gym-action={idx}: dec={float(decision[idx]):.3f} adv={advantage:.3f}"
            ))
        
        # 规则 2: Epsilon-greedy 探索
        if np.random.random() < self.epsilon:
            explore_idx = np.random.randint(0, self.n_actions)
            candidates.append(RuleResult(
                action=candidates[explore_idx].action,
                confidence=0.2,  # 低置信度 = 探索
                explanation=f"Explore: random-action={explore_idx}"
            ))
        
        return candidates
    
    def select_best(self, ctx: RuleContext, candidates: List[RuleResult]) -> RuleResult:
        # 多数时间选最高 Value，偶尔探索
        exploit_candidates = [c for c in candidates if "random" not in c.explanation]
        explore_candidates = [c for c in candidates if "random" in c.explanation]
        
        if explore_candidates and np.random.random() < self.epsilon:
            return explore_candidates[0]
        if exploit_candidates:
            return max(exploit_candidates, key=lambda c: c.confidence)
        return candidates[0]


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
        
        while not done and ep_steps < 500:
            x = obs_to_input(obs)
            trace = reasoner.reason(x, max_steps=3, confidence_threshold=0.5)
            
            # 提取动作
            if trace.final_action is not None:
                action = int(np.argmax(trace.final_action.flatten()[:n_actions]))
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
        
        result.neuroflow_rewards.append(ep_reward)
        nf_total_steps += ep_steps
        
        if (ep + 1) % 10 == 0:
            avg = np.mean(result.neuroflow_rewards[-10:])
            print(f"    Ep {ep+1:3d}/{episodes} | avg_reward={avg:.1f} | best={result.nf_best:.0f}")
    
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
