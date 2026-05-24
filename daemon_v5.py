"""
NeuroFlow v5.0 守护进程 — 子词版
==================================
架构: BPE嵌入 → PE → 因果窗口 → SAE → 记忆 → 输出头
运行: CPU-only, 纯NumPy, 持续自进化
"""
import sys, os, time, json, random, numpy as np
from datetime import datetime
from collections import deque

sys.path.insert(0, "/mnt/d/neuroflow-model")
DEPLOY_PATH = "/mnt/d/neuroflow-model"
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_v5_state.json")
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_v5_weights.npz"
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")
VOCAB_FILE = os.path.join(DEPLOY_PATH, "tokenizer_v5.json")

BATCH_SIZE = 400
MAX_SEQ_LEN = 128        # 每个样本最多128 tokens
EVOLVE_INTERVAL = 16000
SAVE_EVERY = 16000
STATUS_INTERVAL = 1800

from ops_v5 import *
from ops_v5 import (AdaptivePositionBlender, EvolutionMonitorV5, 
                    diagnose_temporal_variance)
from tokenizer_v5 import get_tokenizer, PAD_ID

os.makedirs(KB_DIR, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "40")


# ═══ 内置知识 ═══
BUILTIN_KNOWLEDGE = [
    "artificial intelligence is transforming the world through machine learning",
    "the transformer architecture uses attention mechanisms for sequence processing",
    "neural networks learn hierarchical representations from data",
    "reinforcement learning enables agents to make decisions through trial and error",
    "natural language processing enables computers to understand human language",
    "computer vision allows machines to interpret and analyze visual information",
    "deep learning models require large amounts of training data",
    "transfer learning leverages pre-trained models for new tasks",
    "gradient descent is the primary optimization algorithm in deep learning",
    "backpropagation computes gradients through the chain rule of calculus",
    "convolutional neural networks are designed for grid-like data such as images",
    "recurrent neural networks process sequential data with hidden states",
    "batch normalization stabilizes training by normalizing layer inputs",
    "dropout prevents overfitting by randomly dropping neurons during training",
    "attention mechanisms allow models to focus on relevant parts of input",
    "embedding layers map discrete tokens to continuous vector spaces",
    "the softmax function converts logits to probability distributions",
    "cross entropy loss measures the difference between two probability distributions",
    "stochastic gradient descent updates parameters using random mini batches",
    "learning rate schedules adjust the step size during training",
    "weight decay applies L2 regularization to prevent overfitting",
    "layer normalization normalizes activations across feature dimensions",
    "residual connections help train deeper networks by skipping layers",
    "self attention computes attention scores between all pairs in a sequence",
    "multi head attention runs multiple attention mechanisms in parallel",
    "positional encoding injects sequence position information into embeddings",
    "masked language modeling predicts masked tokens in a sequence",
    "next sentence prediction determines if two sentences are consecutive",
    "fine tuning adapts a pre trained model to a specific downstream task",
    "zero shot learning enables models to handle unseen tasks without examples",
    "few shot learning uses a small number of examples to adapt to new tasks",
    "knowledge distillation transfers knowledge from large to small models",
    "mixture of experts activates only a subset of parameters for each input",
    "sparse autoencoders learn compressed representations with limited activations",
    "memory augmented networks use external memory for long term存储",
    "graph neural networks operate on graph structured data",
    "variational autoencoders learn latent representations with probabilistic constraints",
    "generative adversarial networks pit two networks against each other",
    "diffusion models generate data by reversing a noise process",
    "contrastive learning pulls similar samples together and pushes apart dissimilar ones",
    "self supervised learning learns representations without human labels",
    "meta learning trains models to learn new tasks quickly",
    "online learning updates models incrementally as new data arrives",
    "ensemble methods combine multiple models for better predictions",
    "bayesian neural networks model uncertainty in predictions",
    "capsule networks use vector neurons to preserve spatial relationships",
    "state space models provide efficient alternatives to transformers",
    "mamba is a selective state space model for efficient sequence modeling",
    "the bpe tokenizer splits text into subword units for vocabulary building",
    "byte level tokenization handles arbitrary text without fixed vocabularies",
]


class NeuroFlowV5Daemon:
    """v5.0 守护进程"""

    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.kb_contents = []
        self.kb_idx = 0
        self.epoch = 0
        self._h_global = np.zeros((1, D_MEM), dtype=np.float32)
        self.pe_table = get_pe_table(MAX_SEQ_LEN, D_MODEL)  # 预计算PE表
        self.pos_blender = AdaptivePositionBlender(D_MODEL, MAX_SEQ_LEN, warmup_steps=1000)
        self.evo_monitor = EvolutionMonitorV5(VOCAB_SIZE, var_target=0.0356)
        self.train_step_counter = 0
        self._recent_losses = deque(maxlen=50)
        self._recent_vars = deque(maxlen=50)
        self._init_model()
        self._load_state()
        self._preload_kb()
        self.last_status = time.time()

    def _init_model(self):
        """初始化v5.0模型"""
        np.random.seed(42)
        scale = np.sqrt(2.0 / D_MODEL)

        # Embedding
        self.W_embed = np.random.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.01

        # Causal Window Gate
        self.W_g = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * scale
        self.b_g = np.zeros(D_MODEL, dtype=np.float32)

        # Memory Bank
        K_init = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32)
        K_init = K_init / (np.linalg.norm(K_init, axis=1, keepdims=True) + 1e-8)
        self.M_K = K_init
        self.M_V = np.random.randn(MEM_SLOTS, D_MEM).astype(np.float32) * 0.01
        self.W_read = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_write = np.random.randn(D_MODEL, MEM_SLOTS).astype(np.float32) * 0.01
        self.W_to_mem = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale

        # Output head (low-rank)
        self.W_proj = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.W_out = np.random.randn(D_MEM, VOCAB_SIZE).astype(np.float32) * 0.01
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)
        self.W_proj_out = self.W_proj.copy()
        self.b_proj_out = np.zeros(D_MEM, dtype=np.float32)

        # Vocab head (two-stage)
        self.V_in = np.random.randn(D_MODEL, D_MEM).astype(np.float32) * scale
        self.V_out = np.random.randn(D_MEM, VOCAB_SIZE).astype(np.float32) * 0.01
        self.V_bias = np.zeros(VOCAB_SIZE, dtype=np.float32)

        # Training state
        self.lr = 0.01
        self.n_updates = 0
        self.total_loss = 0.0
        self.state = {
            "started": datetime.now().isoformat(),
            "topics": 0, "evolutions": 0, "fitness": 0.0,
            "errors": 0, "total_loss": 0.0, "train_steps": 0,
            "auto_evolutions": 0,
            "evo_stats": {"auto_tune": 0, "contrastive": 0, "noise_inject": 0,
                          "degrade_lr": 0, "deep_stuck": 0, "w_embed": 0},
        }

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f: self.state.update(json.load(f))
            except: pass

    def _save_state(self):
        self.state["total_loss"] = float(self.total_loss)
        self.state["train_steps"] = int(self.n_updates)
        with open(STATE_FILE, "w") as f: json.dump(self.state, f)

    def _save_weights(self):
        weights = {
            "W_embed": self.W_embed, "W_g": self.W_g, "b_g": self.b_g,
            "M_K": self.M_K, "M_V": self.M_V,
            "W_read": self.W_read, "W_write": self.W_write, "W_to_mem": self.W_to_mem,
            "W_proj": self.W_proj, "W_out": self.W_out, "b_out": self.b_out,
            "V_in": self.V_in, "V_out": self.V_out, "V_bias": self.V_bias,
        }
        np.savez_compressed(WEIGHTS_FILE, **weights)

    def _preload_kb(self):
        """加载KB文件（复用v4逻辑）"""
        t0 = time.time()
        kb_dir = KB_DIR
        if not os.path.isdir(kb_dir): return
        files = sorted(os.listdir(kb_dir), reverse=True)[:5000]  # v5.0: 最多5K条（快速启动）
        contents = []
        for fname in files:
            path = os.path.join(kb_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read(4000).strip()
                if len(text) > 20: contents.append(text)
            except: pass
        self.kb_contents = contents
        print(f"  📖 加载 {len(contents)} 条知识 ({time.time()-t0:.0f}s)")

    def _get_next_text(self) -> str:
        """获取下一条训练文本"""
        # 70% KB, 30% BUILTIN
        if not self.kb_contents or random.random() < 0.3:
            return random.choice(BUILTIN_KNOWLEDGE)
        if self.kb_idx >= len(self.kb_contents):
            self.kb_idx = 0
            self.epoch += 1
            random.shuffle(self.kb_contents)
        text = self.kb_contents[self.kb_idx]
        self.kb_idx += 1
        return text

    def forward(self, token_ids):
        """前向传播（含自适应PE淡入 + 时序方差诊断）"""
        N, L = token_ids.shape
        X_embed = self.W_embed[token_ids]  # (N, L, D)
        
        # Adaptive PE blending per sample (avoid spatial bullying)
        X_list = []
        for i in range(N):
            X_i, lambda_t = self.pos_blender.blend(X_embed[i], self.train_step_counter)
            X_list.append(X_i)
        X = np.stack(X_list, axis=0)

        C = np.zeros_like(X)
        for i in range(N):
            C[i] = causal_window_gating_operator(X[i], self.W_g, self.b_g)

        h = C.reshape(-1, D_MODEL)
        h_abs = np.abs(h)
        K = min(65, h.shape[1])
        thresh = np.partition(h_abs, -K, axis=1)[:, -K:-K+1]
        mask = (h_abs >= thresh).astype(np.float32)
        h = h * mask

        h_pooled = h.reshape(N, L, D_MODEL).mean(axis=1)
        mem_out, _ = memory_read_write(h_pooled[:1], self.M_V,
                                        self.W_read, self.W_write, self.W_to_mem)
        mem_feat = mem_out @ self.W_to_mem.T
        h = h + 0.1 * np.tile(mem_feat, (N * L, 1))

        h_proj = np.maximum(h @ self.W_proj, 0)
        logits = h_proj @ self.W_out + self.b_out

        # h_var for monitoring
        h_var = float(np.var(h))

        return logits, h, h_var

    def train_step(self, texts):
        """训练一步"""
        N = min(BATCH_SIZE, len(texts))
        lr = self.lr

        # Encode texts
        token_ids_list = []
        target_ids_list = []
        for text in texts[:N]:
            ids = self.tokenizer.encode(text, MAX_SEQ_LEN)
            token_ids_list.append(ids)
            # Targets: shift by 1 (predict next token)
            target_ids = np.roll(ids, -1)
            target_ids[-1] = PAD_ID
            target_ids_list.append(target_ids)

        token_ids = np.stack(token_ids_list, axis=0)
        targets = np.stack(target_ids_list, axis=0).ravel()

        # Forward
        logits, h, h_var = self.forward(token_ids)

        # CE Loss
        loss = cross_entropy_loss(logits, targets)
        grad = vocab_gradient(logits, targets)

        # Backward to output head
        h_proj = np.maximum(h @ self.W_proj, 0)
        d_proj = (grad @ self.W_out.T) * (h_proj > 0).astype(np.float32)

        self.W_out -= lr * (h_proj.T @ grad + 0.002 * self.W_out)
        self.b_out -= lr * np.sum(grad, axis=0)
        self.W_proj -= lr * (h.T @ d_proj + 0.002 * self.W_proj)

        self.n_updates += 1
        self.total_loss += loss
        self.train_step_counter += 1
        self._recent_losses.append(loss)
        self._recent_vars.append(h_var)
        return loss, h_var, logits, h

    def run(self):
        """主循环"""
        print(f"[{datetime.now():%H:%M:%S}] 🧠 NeuroFlow v5.0 子词版启动")
        print(f"[{datetime.now():%H:%M:%S}] 📚 知识: 内置{len(BUILTIN_KNOWLEDGE)}条 + {len(self.kb_contents)}文件")
        print(f"[{datetime.now():%H:%M:%S}] ⚙️  batch={BATCH_SIZE} | seq={MAX_SEQ_LEN} | dim={D_MODEL} | vocab={VOCAB_SIZE}")

        batch_count = 0
        while True:
            try:
                texts = [self._get_next_text() for _ in range(BATCH_SIZE)]
                loss, h_var, logits, h = self.train_step(texts)
                self.state["topics"] += len(texts)
                batch_count += 1

                # Evolution
                if self.state["topics"] % EVOLVE_INTERVAL < BATCH_SIZE and self.state["topics"] > 0:
                    # v5.0 fitness: Root-RIG + PRR + Recon + Var
                    recent_loss = np.mean(self._recent_losses) if self._recent_losses else loss
                    recent_var = np.mean(self._recent_vars) if self._recent_vars else h_var
                    fit_val, evo_metrics = self.evo_monitor.evaluate(
                        recon_loss=loss,  # CE loss serves as recon proxy
                        vocab_loss=recent_loss,
                        current_var=recent_var
                    )
                    self.state["fitness"] = fit_val
                    self.state["evolutions"] += 1
                    print(f"  🧬 进化 #{self.state['evolutions']}: "
                          f"fit={fit_val:.4f} (rig={evo_metrics['root_rig']:.3f} "
                          f"prr={evo_metrics['prr']:.3f}) "
                          f"loss={loss:.4f} batch={batch_count}", flush=True)

                # Save
                if self.state["topics"] % SAVE_EVERY < BATCH_SIZE:
                    self._save_weights()
                    self._save_state()

                # Log
                if batch_count % 10 == 0:
                    bar_len = min(30, int(self.state["fitness"] * 30))
                    bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
                    # LVR诊断
                    lvr, v_spec = diagnose_temporal_variance(h.reshape(-1, 128, D_MODEL)) if h.size >= 128 else (0, [])
                    lambda_t = self.pos_blender.blend(self.W_embed[:1], self.train_step_counter)[1]
                    print(f"  [{self.state['topics']:7d}] 📦 batch#{batch_count} e{self.epoch} "
                          f"loss={loss:.4f} var={h_var:.4f} λ={lambda_t:.2f} "
                          f"lvr={lvr:.3f} fit={self.state['fitness']:.4f} | {bar}", flush=True)

                # Status
                now = time.time()
                if now - self.last_status >= STATUS_INTERVAL:
                    self._report_status()
                    self.last_status = now

            except Exception as e:
                print(f"  ❌ 错误: {e}", flush=True)
                self.state["errors"] += 1

    def _report_status(self):
        avg_loss = self.total_loss / max(self.n_updates, 1)
        print(f"\n{'='*50}", flush=True)
        print(f"📊 v5.0 状态报告 | {datetime.now():%Y-%m-%d %H:%M}", flush=True)
        print(f"  步数: {self.n_updates} | 进化: {self.state['evolutions']}", flush=True)
        print(f"  平均Loss: {avg_loss:.4f} | Fit: {self.state['fitness']:.4f}", flush=True)
        print(f"  Epoch: {self.epoch} | KB: {len(self.kb_contents)}条", flush=True)
        print(f"{'='*50}\n", flush=True)


if __name__ == "__main__":
    d = NeuroFlowV5Daemon()
    d.run()
