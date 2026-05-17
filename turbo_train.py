"""
NeuroFlow 涡轮加速训练器 💨
============================
批量处理知识库文件，最大化40核Xeon利用率
无 daemon 开销，纯计算模式
可和 daemon 并行运行（共享权重文件）

用法:
  OMP_NUM_THREADS=40 python3 turbo_train.py
"""

import sys, os, time, json, random, glob
import numpy as np
from datetime import datetime

sys.path.insert(0, "/mnt/d/neuroflow-model")

DEPLOY_PATH = "/mnt/d/neuroflow-model"
KB_DIR = os.path.join(DEPLOY_PATH, "knowledge_base")
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights_v4.npz"
STATE_FILE = os.path.join(DEPLOY_PATH, "daemon_state.json")

TEXT_DIM = 512
HIDDEN_DIM = 256
OUTPUT_DIM = 10
HEAD_ACTIONS = 10
BATCH_SIZE = 8       # 每批次并行处理8条

os.environ.setdefault("OMP_NUM_THREADS", "40")


def encode_batch(texts: list[str], dim: int = 512) -> np.ndarray:
    """批量编码 — 向量化实现，充分利用CPU"""
    batch = []
    for text in texts:
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        n_words = min(len(words), 500)
        for i, word in enumerate(words[:n_words]):
            h = abs(hash(word)) % (2**31)
            for j in range(8):
                idx = (h + j * 2654435761) % dim
                vec[int(idx)] += 0.03 / max(n_words / 30, 1)
        vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
        norms = np.linalg.norm(vec)
        if norms > 1e-8:
            vec /= norms
        batch.append(vec)
    return np.array(batch, dtype=np.float32)


def load_knowledge_texts(kb_dir: str, max_files: int = 0) -> list[str]:
    """从知识库提取所有文本条目"""
    texts = []
    files = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    if max_files > 0:
        files = files[:max_files]
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read(5000).strip()
            if len(content) > 30:
                texts.append(content)
        except:
            pass
    return texts


def turbo_train():
    print(f"[{datetime.now():%H:%M:%S}] 🚀 NeuroFlow 涡轮加速训练启动")
    print(f"[{datetime.now():%H:%M:%S}] 📚 加载知识库...")
    
    texts = load_knowledge_texts(KB_DIR)
    print(f"[{datetime.now():%H:%M:%S}] 📚 已加载 {len(texts)} 条知识")
    
    print(f"[{datetime.now():%H:%M:%S}] 🧠 加载模型...")
    from neuroflow._core import create_multimodal
    from neuroflow.cognition import NeuroSymbolicReasoner
    from neuroflow.trainable_head import TrainableHead
    
    model = create_multimodal(
        text_dim=TEXT_DIM, image_size=224, output_dim=OUTPUT_DIM, quantize=True
    )
    reasoner = NeuroSymbolicReasoner(model)
    head = TrainableHead(
        model, hidden_dim=HIDDEN_DIM, n_actions=HEAD_ACTIONS, lr=0.01
    )
    
    # 加载已有权重
    if os.path.exists(WEIGHTS_FILE):
        try:
            data = np.load(WEIGHTS_FILE)
            head.load_weights({
                "W_d": data.get("W_d", head.W_d),
                "b_d": data.get("b_d", head.b_d),
                "W_v": data.get("W_v", head.W_v),
                "b_v": data.get("b_v", head.b_v),
            })
            print(f"[{datetime.now():%H:%M:%S}] ✅ 已加载已有权重")
        except:
            print(f"[{datetime.now():%H:%M:%S}] ⚠️ 权重加载失败，从头训练")
    
    # 加载状态（获取已学主题数）
    topics_done = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            topics_done = saved.get("topics", 0)
        except:
            pass
    
    t_start = time.time()
    total_loss = 0.0
    train_steps = 0
    
    # 打乱数据，确保多样性
    random.shuffle(texts)
    
    batch_idx = 0
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"[{datetime.now():%H:%M:%S}] ⚡ 开始批量训练 ({n_batches} batches × {BATCH_SIZE})")
    print(f"[{datetime.now():%H:%M:%S}] 💪 CPU: {os.environ.get('OMP_NUM_THREADS', '40')} 线程")
    print()
    
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        batch_idx += 1
        
        # 1. 批量编码
        X = encode_batch(batch_texts)
        
        # 2. 批量推理
        for i in range(len(batch_texts)):
            x = X[i:i+1]  # 保持2D
            trace = reasoner.reason(x, max_steps=5)
            
            conf = trace.final_confidence if trace.steps else 0.5
            reward = conf
            
            if trace.final_action is not None and trace.total_steps > 0:
                target = int(np.argmax(trace.final_action))
                result = head.train_step(x, target, reward)
                total_loss += result["loss"]
                train_steps += 1
                topics_done += 1
        
        # 进度
        if batch_idx % 5 == 0 or batch_idx == n_batches:
            elapsed = time.time() - t_start
            rate = train_steps / elapsed if elapsed > 0 else 0
            avg_loss = total_loss / max(train_steps, 1)
            pct = min(100, batch_idx * BATCH_SIZE / len(texts) * 100)
            
            print(f"  [{batch_idx:4d}/{n_batches}] "
                  f"{pct:5.1f}% | "
                  f"{rate:.1f} 条/秒 | "
                  f"loss={avg_loss:.4f} | "
                  f"主题={topics_done}", flush=True)
        
        # 定期保存权重
        if batch_idx % 20 == 0:
            w = head.get_weights()
            np.savez(WEIGHTS_FILE, **w)
    
    # 最终保存
    w = head.get_weights()
    np.savez(WEIGHTS_FILE, **w)
    
    elapsed = time.time() - t_start
    avg_loss = total_loss / max(train_steps, 1)
    rate = train_steps / elapsed if elapsed > 0 else 0
    
    print()
    print(f"[{datetime.now():%H:%M:%S}] {'='*50}")
    print(f"[{datetime.now():%H:%M:%S}] ✅ 涡轮训练完成！")
    print(f"[{datetime.now():%H:%M:%S}] 📊 处理: {train_steps} 条 | {elapsed:.1f} 秒")
    print(f"[{datetime.now():%H:%M:%S}] 📊 速度: {rate:.1f} 条/秒")
    print(f"[{datetime.now():%H:%M:%S}] 📊 损失: {avg_loss:.4f}")
    print(f"[{datetime.now():%H:%M:%S}] 📊 主题: {topics_done}")


if __name__ == "__main__":
    turbo_train()
