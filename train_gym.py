"""
NeuroFlow Gym 在线 RL 训练 v2 — REINFORCE 模式
==============================================
修复: 每步 reward=1 无区分度 → 整局结束后用总回报训练

REINFORCE 算法:
  收集整局 (state, action) → episode_reward
  对每步训练: loss *= (reward - baseline) 缩放梯度
"""
import sys, os, time, json, numpy as np

sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import create_multimodal
from neuroflow.trainable_head import TrainableHead

WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights.npz"
RESULT_FILE = "/mnt/d/neuroflow-model/benchmarks/cartpole_trained_result.json"

def obs_to_input(observation, input_dim=512):
    obs = np.array(observation, dtype=np.float32)
    obs_norm = obs / (np.abs(obs).max() + 1e-8)
    repeats = input_dim // len(obs_norm) + 1
    x = np.tile(obs_norm, repeats)[:input_dim].astype(np.float32)
    return x.reshape(1, -1)

def evaluate(head, env, episodes=20):
    """纯评估：epsilon=0"""
    n_actions = env.action_space.n
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        while not done and steps < 500:
            x = obs_to_input(obs)
            pred = head.predict(x)
            action = int(np.argmax(pred.decision[0])) % n_actions
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        rewards.append(ep_reward)
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.max(rewards))


def train_step_reinforce(head, x, action, advantage):
    """
    REINFORCE 单步训练: 用 advantage 缩放决策梯度。
    
    正 advantage → 增强该动作概率
    负 advantage → 抑制该动作概率
    """
    batch = x.shape[0]
    
    # 前传
    head._forward(x)
    h = head._last_hidden.astype(np.float32)
    
    decision = h @ head.W_d + head.b_d
    value = h @ head.W_v + head.b_v
    
    # Softmax
    d_shifted = decision - np.max(decision, axis=1, keepdims=True)
    exp_d = np.exp(d_shifted)
    probs = exp_d / (np.sum(exp_d, axis=1, keepdims=True) + 1e-8)
    
    # 损失
    target_onehot = np.zeros_like(decision)
    target_onehot[0, action] = 1.0
    
    ce_loss = -np.log(probs[0, action] + 1e-8)
    value_loss = (value[0, 0] - advantage) ** 2
    loss = ce_loss + 0.1 * value_loss
    
    # REINFORCE 梯度 (advantage 缩放)
    # 正 advantage → 梯度推高 action 概率
    # 负 advantage → 梯度推低 action 概率
    grad_d_logits = (probs - target_onehot) * advantage / batch
    grad_W_d = h.T @ grad_d_logits
    grad_b_d = np.sum(grad_d_logits, axis=0, keepdims=True)
    
    # 价值梯度
    grad_v = 2 * (value - advantage) / batch
    grad_W_v = h.T @ grad_v
    grad_b_v = np.sum(grad_v, axis=0, keepdims=True)
    
    # SGD
    head.W_d -= head.lr * grad_W_d
    head.b_d -= head.lr * grad_b_d
    head.W_v -= head.lr * grad_W_v * 0.1
    head.b_v -= head.lr * grad_b_v * 0.1
    
    head.n_updates += 1
    head.total_loss += float(loss)
    
    return {"loss": float(loss), "advantage": advantage}


def train(env_name="CartPole-v1", episodes=300, epsilon=0.15, lr=0.03):
    try:
        import gymnasium as gym
    except ImportError:
        os.system(f"{sys.executable} -m pip install gymnasium -q")
        import gymnasium as gym

    env = gym.make(env_name)
    n_actions = env.action_space.n
    max_steps = 500

    model = create_multimodal(text_dim=512, image_size=224, output_dim=10, quantize=False)
    head = TrainableHead(model, hidden_dim=256, n_actions=10, lr=lr)

    if os.path.exists(WEIGHTS_FILE):
        try:
            weights = dict(np.load(WEIGHTS_FILE, allow_pickle=True))
            head.load_weights(weights)
        except:
            pass

    # 训练前评估
    pre_avg, pre_std, pre_best = evaluate(head, env, episodes=10)
    random_baseline = 23.1  # CartPole random baseline
    print(f"\n📊 训练前: avg={pre_avg:.1f} ±{pre_std:.1f}  best={pre_best:.0f}")
    print(f"{'='*60}")

    t0 = time.time()
    all_rewards = []
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        transitions = []  # (obs_vec, action)
        ep_reward = 0
        done = False
        steps = 0

        # === 收集轨迹 ===
        while not done and steps < max_steps:
            x = obs_to_input(obs)
            pred = head.predict(x)
            
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(pred.decision[0])) % n_actions
            
            transitions.append((x.copy(), action))
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            total_steps += 1

        all_rewards.append(ep_reward)

        # === REINFORCE: 整局结束后训练 ===
        # advantage = (ep_reward - baseline) / max_steps  归一化到 [-1, 1]
        advantage = (ep_reward - random_baseline) / max_steps
        
        for x, action in transitions:
            train_step_reinforce(head, x, action, advantage)

        # 进度
        if (ep + 1) % 10 == 0:
            recent = np.mean(all_rewards[-10:])
            stats = head.stats()
            print(f"  Ep {ep+1:3d}/{episodes} | "
                  f"reward={recent:6.1f} | "
                  f"adv={advantage:+.4f} | "
                  f"W_d={stats['W_d_norm']:.2f} b_d={stats['b_d_norm']:.2f} | "
                  f"best={np.max(all_rewards):.0f}")

    env.close()
    elapsed = time.time() - t0

    # 训练后评估
    env2 = gym.make(env_name)
    post_avg, post_std, post_best = evaluate(head, env2, episodes=30)
    env2.close()

    improvement = (post_avg - pre_avg) / max(abs(pre_avg), 1) * 100

    print(f"\n{'='*60}")
    print(f"🏁 训练完成 {elapsed:.1f}s")
    print(f"  训练前: avg={pre_avg:7.1f} ±{pre_std:5.1f}  best={pre_best:.0f}")
    print(f"  训练后: avg={post_avg:7.1f} ±{post_std:5.1f}  best={post_best:.0f}")
    print(f"  提升:   {improvement:+.1f}%")
    print(f"  总步数: {total_steps} | 最后10轮 avg: {np.mean(all_rewards[-10:]):.1f}")

    np.savez_compressed(WEIGHTS_FILE, **head.get_weights())
    print(f"\n💾 权重已保存: {WEIGHTS_FILE}")

    result = {
        "env": env_name, "episodes": episodes, "total_steps": total_steps,
        "lr": lr, "epsilon": epsilon, "method": "REINFORCE",
        "pre_train": {"avg": pre_avg, "std": pre_std, "best": pre_best},
        "post_train": {"avg": post_avg, "std": post_std, "best": post_best},
        "improvement_pct": round(improvement, 1),
        "reward_curve": all_rewards,
        "elapsed_s": round(elapsed, 1),
        "head_stats": head.stats(),
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=0.03)
    args = p.parse_args()
    train(episodes=args.episodes, epsilon=args.epsilon, lr=args.lr)
