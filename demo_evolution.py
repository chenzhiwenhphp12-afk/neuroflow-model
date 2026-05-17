"""
NeuroFlow 自主进化 Demo
=======================
演示：链式推理 + 经验学习 + 自主进化
"""

import sys, numpy as np, time
sys.path.insert(0, "/mnt/d/neuroflow-model")

from neuroflow._core import NeuroFlowLite
from neuroflow.cognition import ReasoningLoop, SelfEvolution, AutonomousAgent

print("=" * 64)
print("  🧠 NeuroFlow 自主进化 Demo")
print("  逻辑推理 + 经验学习 + 优胜劣汰 = 智能涌现")
print("=" * 64)

# 初始化
model = NeuroFlowLite(input_dim=512)
agent = AutonomousAgent(model, name="NF-1")

print(f"\n📋 智能体: {agent.name}")
print(f"  模型: NeuroFlow Lite (331K params)")
print(f"  模块: ReasoningLoop + SelfEvolution")
print()

# ============================================================
# 阶段 1: 链式推理演示
# ============================================================
print("=" * 64)
print("  🔗 阶段 1: 链式推理 — 多步思考过程")
print("=" * 64)

x = np.random.randn(1, 512).astype(np.float32) * 0.1

print("\n  输入: 随机信号 (模拟外部刺激)")
trace = agent.reasoner.reason(x, max_steps=8, confidence_threshold=0.0, verbose=True)
print(f"\n  推理完成: {trace.total_steps} 步, {trace.convergence_time:.1f}ms")
print(f"  最终决策: {trace.final_decision[0,:5].round(3)}")

# ============================================================
# 阶段 2: 经验学习
# ============================================================
print("\n" + "=" * 64)
print("  📚 阶段 2: 经验积累 — 从环境反馈中学习")
print("=" * 64)

env_patterns = {
    "safe":    (0.9, "高奖励 — 安全环境"),
    "neutral": (0.5, "中奖励 — 普通环境"),
    "danger":  (0.1, "低奖励 — 危险环境"),
}

for pattern, (reward, desc) in env_patterns.items():
    print(f"\n  [{desc}]")
    for i in range(10):
        x = np.random.randn(1, 512).astype(np.float32) * 0.1
        # 根据环境调整输入特征
        if pattern == "danger":
            x *= 2.0  # 高幅度 = 危险信号
        elif pattern == "safe":
            x *= 0.5  # 低幅度 = 安全信号

        agent.learn_from_feedback(x, reward=reward)

    reflect = agent.evolution.reflect()
    print(f"    经验数: {reflect['n_experiences']}, "
          f"平均奖励: {reflect.get('mean_reward', 0):.3f}")

# ============================================================
# 阶段 3: 自主进化
# ============================================================
print("\n" + "=" * 64)
print("  🧬 阶段 3: 自主进化 — 优胜劣汰")
print("=" * 64)

print("\n  Before evolution:")
fitness_before = agent.evolution.get_fitness()
print(f"    最佳适应度: {fitness_before['best_fitness']:.4f}")

print("\n  Evolving...")
t0 = time.time()
agent.evolution.evolve(generations=50, population_size=5, verbose=True)
print(f"\n  Evolution completed in {time.time()-t0:.1f}s")

print("\n  After evolution:")
fitness_after = agent.evolution.get_fitness()
print(f"    最佳适应度: {fitness_after['best_fitness']:.4f}")
print(f"    平均适应度: {fitness_after['current_avg_fitness']:.4f}")
print(f"    总经验:     {fitness_after['total_experiences']}")

# ============================================================
# 阶段 4: 自动课程学习
# ============================================================
print("\n" + "=" * 64)
print("  📈 阶段 4: 自动课程学习 — 从简单到复杂")
print("=" * 64)

print("\n  生成课程 (递增难度)...")
for step in range(5):
    agent.evolution.auto_curriculum(steps=5)
    status = agent.get_status()
    fit = status['fitness']
    print(f"    课程 {step+1}: 经验={status['memory_size']:4d}, "
          f"适应度={fit['best_fitness']:.4f}, "
          f"平均奖励={fit['avg_reward']:.3f}")

# ============================================================
# 阶段 5: 记忆巩固
# ============================================================
print("\n" + "=" * 64)
print("  💤 阶段 5: 记忆巩固 — 模拟睡眠 LTP")
print("=" * 64)

print("\n  Consolidating high-reward memories...")
agent.evolution.consolidate()
print(f"  ✓ 记忆巩固完成")
print(f"    经验总数: {len(agent.evolution.experience_buffer)}")
print(f"    最佳适应度: {agent.evolution.best_fitness:.4f}")

# ============================================================
# 适应度曲线
# ============================================================
print("\n" + "=" * 64)
print("  📊 进化适应度曲线")
print("=" * 64)

hist = agent.evolution.fitness_history
if hist:
    step = max(1, len(hist) // 15)
    for i in range(0, len(hist), step):
        bar = "█" * int(hist[i] * 30)
        print(f"    Gen {i:4d}: {hist[i]:.4f} {bar}")

# ============================================================
# 最终状态
# ============================================================
print("\n" + "=" * 64)
print("  🏁 最终智能体状态")
print("=" * 64)

status = agent.get_status()
for k, v in status.items():
    if k == 'fitness':
        for fk, fv in v.items():
            if fk != 'fitness_trend':
                print(f"  {k}.{fk}: {fv}")
    else:
        print(f"  {k}: {v}")

print("\n" + "=" * 64)
print("  ✅ 自主进化 Demo 完成")
print("  🧠 链式推理 ✓  经验学习 ✓  优胜劣汰 ✓  课程学习 ✓")
print("=" * 64)
