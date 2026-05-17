"""
CartPole 线性策略梯度训练器
============================
独立轻量实现，不依赖 NeuroFlow C++ 模块。
使用简单线性 softmax 策略 + REINFORCE。

预期: 500-1000 episodes 内收敛到 CartPole-v1 的 200+

启动: nohup python3 train_cartpole_linear.py &> cartpole_linear.log &
"""

import sys, os, time, json, numpy as np
from datetime import datetime
from collections import deque

try:
    import gymnasium as gym
except ImportError:
    os.system(f"{sys.executable} -m pip install gymnasium -q")
    import gymnasium as gym

# ── 配置 ──
STATE_DIM   = 4       # CartPole: [pos, vel, angle, angular_vel]
N_ACTIONS   = 2       # 左/右
LR          = 0.01    # 学习率
GAMMA       = 0.99    # 折扣因子
BATCH_SIZE  = 5       # 每批 episode 数
LOG_EVERY   = 10      # 每 N episode 打印
SAVE_EVERY  = 100     # 每 N episode 保存

WEIGHTS_FILE = "/home/administrator/.hermes/cartpole_linear_weights.npz"
LOG_FILE     = "/mnt/d/neuroflow-model/cartpole_linear_state.json"


class LinearPolicy:
    """线性 softmax 策略: π(a|s) = softmax(W @ s + b)"""

    def __init__(self):
        # Xavier init
        scale = np.sqrt(2.0 / STATE_DIM)
        self.W = np.random.randn(STATE_DIM, N_ACTIONS).astype(np.float64) * scale
        self.b = np.zeros(N_ACTIONS, dtype=np.float64)

    def forward(self, state):
        """返回动作概率"""
        s = np.asarray(state, dtype=np.float64)
        logits = s @ self.W + self.b
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / exp.sum()

    def sample(self, state):
        """按概率采样动作"""
        probs = self.forward(state)
        return int(np.random.choice(N_ACTIONS, p=probs)), probs

    def update(self, states, actions, returns):
        """
        REINFORCE 策略梯度更新。
        ∂J/∂θ = Σ_t γ^t * G_t * ∇log π(a_t|s_t)
        """
        n = len(states)
        if n == 0:
            return 0.0

        total_loss = 0.0
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for t in range(n):
            s = np.asarray(states[t], dtype=np.float64)
            a = actions[t]
            G = returns[t]

            probs = self.forward(s)
            log_prob = np.log(max(probs[a], 1e-8))
            total_loss += -log_prob * G

            # ∂log π(a|s)/∂logits_j = 1_{j=a} - π(j|s)
            grad_logits = -probs.copy()
            grad_logits[a] += 1.0
            grad_logits *= G

            grad_W += np.outer(s, grad_logits)
            grad_b += grad_logits

        # 平均梯度 + SGD
        self.W -= LR * grad_W / n
        self.b -= LR * grad_b / n

        return total_loss / n


def compute_returns(rewards, gamma=GAMMA):
    """折扣回报 + z-score 标准化"""
    T = len(rewards)
    G = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    if G.std() > 1e-8:
        G = (G - G.mean()) / G.std()
    return G


def run_forever():
    env = gym.make("CartPole-v1")
    policy = LinearPolicy()

    # 加载已有权重
    if os.path.exists(WEIGHTS_FILE):
        try:
            data = np.load(WEIGHTS_FILE)
            policy.W = data["W"]
            policy.b = data["b"]
            print(f"[{datetime.now():%H:%M:%S}] 📦 加载权重")
        except:
            pass

    total_ep = 0
    best_reward = 0.0
    recent = deque(maxlen=100)

    print(f"[{datetime.now():%H:%M:%S}] 🎯 CartPole 线性策略训练")
    print(f"[{datetime.now():%H:%M:%S}] ⚙️  lr={LR} γ={GAMMA} batch={BATCH_SIZE}")

    while True:
        try:
            # 收集 BATCH_SIZE 个 episode
            all_states, all_actions, all_returns = [], [], []
            ep_rewards = []

            for _ in range(BATCH_SIZE):
                obs, _ = env.reset()
                done = False
                states, actions, rewards = [], [], []

                while not done:
                    action, probs = policy.sample(obs)
                    states.append(obs.copy())
                    actions.append(action)

                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    rewards.append(float(reward))

                total_ep += 1
                ep_r = sum(rewards)
                ep_rewards.append(ep_r)
                recent.append(ep_r)
                if ep_r > best_reward:
                    best_reward = ep_r

                # 计算回报
                G = compute_returns(rewards)
                all_states.extend(states)
                all_actions.extend(actions)
                all_returns.extend(G.tolist())

            # 策略梯度更新
            loss = policy.update(all_states, all_actions, all_returns)

            # 日志
            if total_ep % LOG_EVERY == 0:
                avg_r = float(np.mean(ep_rewards))
                roll_r = float(np.mean(recent)) if recent else 0
                bar = "█" * min(30, int(avg_r / 16)) + "░" * max(0, 30 - int(avg_r / 16))
                print(f"  [Ep {total_ep:5d}] rew={avg_r:6.1f} best={best_reward:6.0f} "
                      f"loss={loss:.4f} roll={roll_r:5.1f} | {bar}", flush=True)

            # 保存
            if total_ep % SAVE_EVERY == 0:
                np.savez(WEIGHTS_FILE, W=policy.W, b=policy.b)
                with open(LOG_FILE, "w") as f:
                    json.dump({
                        "episodes": total_ep,
                        "best_reward": float(best_reward),
                        "updated": datetime.now().isoformat(),
                    }, f, indent=2)

            # 达标
            if avg_r >= 450:
                print(f"\n  🎉 已掌握 CartPole！avg={avg_r:.0f}")
                np.savez(WEIGHTS_FILE, W=policy.W, b=policy.b)
                break

        except KeyboardInterrupt:
            print(f"\n[{datetime.now():%H:%M:%S}] 💾 保存退出")
            np.savez(WEIGHTS_FILE, W=policy.W, b=policy.b)
            break
        except Exception as e:
            print(f"  ⚠️ {e}", flush=True)
            time.sleep(3)

    env.close()


if __name__ == "__main__":
    run_forever()
