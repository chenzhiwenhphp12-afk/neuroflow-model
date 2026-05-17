"""
CartPole 在线 RL 训练 v2 — 使用 TrainableHead 策略梯度
========================================================
在 CartPole-v1 环境中持续运行，使用 REINFORCE 训练 TrainableHead。

策略: state → model.forward_text → head.predict → softmax → 采样动作
训练: 收集 episode 轨迹 → 计算回报 → head.train_batch()

启动: nohup python3 train_cartpole_rl.py &> cartpole_rl.log &
"""

import sys, os, time, json, random, numpy as np
from datetime import datetime
from collections import deque

sys.path.insert(0, "/mnt/d/neuroflow-model")

# ── 配置 ──
DEPLOY_PATH     = "/mnt/d/neuroflow-model"
WEIGHTS_FILE    = "/home/administrator/.hermes/neuroflow_weights_rl.npz"
LOG_FILE        = os.path.join(DEPLOY_PATH, "cartpole_rl_state.json")
TEXT_DIM        = 512
HIDDEN_DIM      = 256
OUTPUT_DIM      = 10
N_ACTIONS       = 2

# RL 超参数
EPISODES_PER_CYCLE = 10
LEARNING_RATE      = 0.02
GAMMA              = 0.99
MAX_EP_STEPS       = 500
SAVE_INTERVAL      = 100
LOG_INTERVAL       = 10

try:
    import gymnasium as gym
except ImportError:
    os.system(f"{sys.executable} -m pip install gymnasium -q")
    import gymnasium as gym

from neuroflow._core import create_multimodal
from neuroflow.trainable_head import TrainableHead


def obs_to_input(observation, dim=TEXT_DIM):
    obs = np.array(observation, dtype=np.float32)
    obs_norm = obs / (np.abs(obs).max() + 1e-8)
    repeats = dim // len(obs_norm) + 1
    return np.tile(obs_norm, repeats)[:dim].astype(np.float32).reshape(1, -1)


def compute_returns(rewards, gamma=GAMMA):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / returns.std()
    return returns


class CartPoleRLv2:
    """CartPole 在线 RL 训练器（TrainableHead 版）"""

    def __init__(self):
        self.model = create_multimodal(
            text_dim=TEXT_DIM, image_size=224,
            output_dim=OUTPUT_DIM, quantize=True
        )
        self.head = TrainableHead(
            self.model, hidden_dim=HIDDEN_DIM,
            n_actions=N_ACTIONS, lr=LEARNING_RATE
        )
        self.env = gym.make("CartPole-v1")

        self.total_episodes = 0
        self.best_reward = 0.0
        self.recent_rewards = deque(maxlen=100)
        self.start_time = datetime.now()

        self._load_weights()

    def _load_weights(self):
        if not os.path.exists(WEIGHTS_FILE):
            return
        try:
            data = np.load(WEIGHTS_FILE)
            self.head.load_weights({
                "W_d": data.get("W_d", self.head.W_d),
                "b_d": data.get("b_d", self.head.b_d),
                "W_v": data.get("W_v", self.head.W_v),
                "b_v": data.get("b_v", self.head.b_v),
            })
            print(f"[{datetime.now():%H:%M:%S}] 📦 加载 RL 权重")
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] ⚠️ 权重加载失败: {e}")

    def _save_weights(self):
        try:
            w = self.head.get_weights()
            np.savez(WEIGHTS_FILE, **w)
        except Exception:
            pass

    def collect_episode(self) -> dict:
        """运行一个 episode，收集 (state, action, reward) 轨迹"""
        obs, _ = self.env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        states, actions, step_rewards = [], [], []

        while not done and steps < MAX_EP_STEPS:
            x = obs_to_input(obs)

            # head.predict 返回 decision[:N_ACTIONS]
            output = self.head.predict(x)
            dec = output.decision.flatten()[:N_ACTIONS].astype(np.float64)

            # Softmax 采样
            dec_shifted = dec - np.max(dec)
            exp_d = np.exp(dec_shifted)
            probs = exp_d / (np.sum(exp_d) + 1e-8)
            action = int(np.random.choice(N_ACTIONS, p=probs))

            states.append(x)
            actions.append(action)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            step_rewards.append(float(reward))
            ep_reward += reward
            steps += 1

        returns = compute_returns(np.array(step_rewards, dtype=np.float32))
        return {
            "states": states, "actions": actions,
            "returns": returns, "reward": ep_reward, "steps": steps,
        }

    def train_on_trajectories(self, trajectories: list) -> float:
        """REINFORCE 策略梯度：手动实现加权梯度更新
        
        对每个 step:
          - G_t > 0 (前一半 episode): 增强所选动作概率
          - G_t < 0 (后一半 episode): 减弱所选动作概率
        梯度缩放因子 = |G_t| (z-scored)
        """
        total_loss = 0.0
        n_samples = 0
        
        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]
            returns = traj["returns"]  # z-scored
            
            n = len(states)
            if n < 2:
                continue
            
            for t in range(n):
                x = states[t]
                a = actions[t]
                G = float(returns[t])
                
                # 前向：获取当前概率分布
                output = self.head.predict(x)
                dec = output.decision.flatten()[:N_ACTIONS].astype(np.float64)
                dec_shifted = dec - np.max(dec)
                exp_d = np.exp(dec_shifted)
                probs = exp_d / (np.sum(exp_d) + 1e-8)
                
                # 损失：加权负对数似然
                log_prob = np.log(max(probs[a], 1e-8))
                loss = -log_prob * G  # REINFORCE
                total_loss += loss
                n_samples += 1
                
                # 手动策略梯度更新
                # ∂L/∂logit_j = probs[j] - 1_{j==a}    (对交叉熵)
                # 加权: grad *= G
                grad_logits = probs.copy()
                grad_logits[a] -= 1.0
                grad_logits *= G * self.head.lr
                
                # 通过 head.W_d 反向传播到权重
                # h = self.head._last_hidden (1, hidden_dim)
                h = self.head._last_hidden.astype(np.float64)
                # 只更新 action 对应的列
                # W_d: (hidden_dim, n_actions)
                # ∂L/∂W_d = h^T @ grad_logits
                grad_W = h.T @ grad_logits.reshape(1, -1)  # (hidden_dim, n_actions)
                grad_b = grad_logits.reshape(1, -1)  # (1, n_actions)
                
                self.head.W_d -= grad_W
                self.head.b_d -= grad_b * self.head.lr
                
                self.head.n_updates += 1
        
        self.head.total_loss += total_loss
        return total_loss / max(n_samples, 1)

    def run_cycle(self) -> dict:
        trajectories = []
        all_rewards = []

        for _ in range(EPISODES_PER_CYCLE):
            traj = self.collect_episode()
            trajectories.append(traj)
            all_rewards.append(traj["reward"])
            self.total_episodes += 1

        loss = self.train_on_trajectories(trajectories)
        avg_reward = float(np.mean(all_rewards))
        self.recent_rewards.append(avg_reward)

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward

        return {
            "episodes": self.total_episodes,
            "avg_reward": avg_reward,
            "best_reward": self.best_reward,
            "loss": loss,
        }

    def run_forever(self):
        print(f"[{datetime.now():%H:%M:%S}] 🎮 CartPole RL v2 (TrainableHead) 启动")
        print(f"[{datetime.now():%H:%M:%S}] 📐 REINFORCE | lr={LEARNING_RATE} γ={GAMMA}")
        print()

        while True:
            try:
                result = self.run_cycle()

                if result["episodes"] % LOG_INTERVAL == 0:
                    rolling = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0
                    bar_len = min(30, int(result["avg_reward"] / 16))
                    bar = "█" * bar_len + "░" * max(0, 30 - bar_len)
                    print(f"  [Ep {result['episodes']:5d}] "
                          f"rew={result['avg_reward']:6.1f} "
                          f"best={result['best_reward']:6.0f} "
                          f"loss={result['loss']:.4f} "
                          f"roll={rolling:5.1f} | {bar}",
                          flush=True)

                if result["episodes"] % SAVE_INTERVAL == 0:
                    self._save_weights()
                    with open(LOG_FILE, "w") as f:
                        json.dump({
                            "episodes": result["episodes"],
                            "best_reward": result["best_reward"],
                            "total_loss": self.head.total_loss,
                            "n_updates": self.head.n_updates,
                            "updated": datetime.now().isoformat(),
                        }, f, indent=2)

                if result["avg_reward"] >= 450:
                    print(f"\n  🎉 CartPole 已掌握！avg={result['avg_reward']:.0f}")
                    self._save_weights()
                    break

            except KeyboardInterrupt:
                print(f"\n[{datetime.now():%H:%M:%S}] ⏸️ 保存并退出...")
                self._save_weights()
                break
            except Exception as e:
                print(f"  ⚠️ 错误: {e}", flush=True)
                time.sleep(5)
                try:
                    self.env.close()
                except:
                    pass
                self.env = gym.make("CartPole-v1")

        self.env.close()


if __name__ == "__main__":
    trainer = CartPoleRLv2()
    trainer.run_forever()
