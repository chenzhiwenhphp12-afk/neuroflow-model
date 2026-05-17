"""检查 NeuroFlow C++ 模型的 forward_text 输出结构"""
import sys; sys.path.insert(0, "/mnt/d/neuroflow-model")
import numpy as np
from neuroflow._core import create_multimodal, ModelOutput

model = create_multimodal(text_dim=1024, image_size=224, output_dim=10, quantize=True)

# 用两个不同输入观察输出是否相同
from neuroflow.cognition import NeuroSymbolicReasoner

# 测试不同输入
texts = [
    "Gravity is a fundamental force of nature",
    "asdf qwerty zxcv bnm poiu",
    "The CPU processes instructions through fetch decode execute cycle",
    "hi",
]

reasoner = NeuroSymbolicReasoner(model)

for i, t in enumerate(texts):
    # Encode
    words = t.lower().split()
    dim = 1024
    vec = np.zeros(dim, dtype=np.float32)
    n_words = min(len(words), 500)
    for j, word in enumerate(words[:n_words]):
        h = abs(hash(word)) % (2**31)
        for k in range(8):
            idx = (h + k * 2654435761) % dim
            vec[int(idx)] += 0.03 / max(n_words / 30, 1)
    vec += np.sin(np.linspace(0, np.pi * n_words / 15, dim)).astype(np.float32) * 0.08
    vec /= (np.linalg.norm(vec) + 1e-8)
    
    x = vec.reshape(1, -1).astype(np.float32)
    
    # Direct model forward
    out = model.forward_text(x)
    
    # Check fields
    dec = out.decision.flatten() if hasattr(out, 'decision') else None
    val = out.value.flatten() if hasattr(out, 'value') else None
    sal = out.saliency.flatten() if hasattr(out, 'saliency') else None
    mem = out.retrieved_mem.flatten() if hasattr(out, 'retrieved_mem') else None
    gates = out.gates.flatten() if hasattr(out, 'gates') else None
    anom = out.anomaly.flatten() if hasattr(out, 'anomaly') else None
    
    print(f"\n--- 文本 #{i}: {t[:50]} ---")
    print(f"  decision  : shape={getattr(out,'decision',None).shape if hasattr(out,'decision') else 'N/A'}")
    if dec is not None: print(f"    top3: {sorted(dec, reverse=True)[:3]}")
    if val is not None: print(f"  value     : {val}")
    if sal is not None: print(f"  saliency  : {sal}")
    if mem is not None: print(f"  retrieved_mem: shape={mem.shape}")
    if gates is not None: print(f"  gates     : shape={gates.shape}")
    if anom is not None: print(f"  anomaly   : {anom}")
    
    # Reason
    trace = reasoner.reason(x, max_steps=5, verbose=False)
    print(f"  reason    : conf={trace.final_confidence:.4f}, steps={len(trace.steps)}")
    if trace.final_action is not None:
        act = trace.final_action.flatten()
        print(f"  final_action: argmax={np.argmax(act)}, top3={sorted(act, reverse=True)[:3]}")

print("\n\n==== 判断：不同输入是否有不同输出？ ====")
# Check if all outputs are the same (degenerate)
all_same = True
first_out = model.forward_text(vec.reshape(1, -1).astype(np.float32))
first_dec = first_out.decision.flatten()
for t in texts[1:]:
    words = t.lower().split()
    v = np.zeros(dim, dtype=np.float32)
    nw = min(len(words), 500)
    for j, w in enumerate(words[:nw]):
        h = abs(hash(w)) % (2**31)
        for k in range(8):
            idx = (h + k * 2654435761) % dim
            v[int(idx)] += 0.03 / max(nw / 30, 1)
    v += np.sin(np.linspace(0, np.pi * nw / 15, dim)).astype(np.float32) * 0.08
    v /= (np.linalg.norm(v) + 1e-8)
    o = model.forward_text(v.reshape(1, -1).astype(np.float32))
    if not np.allclose(first_dec, o.decision.flatten(), atol=1e-5):
        all_same = False
        break

print(f"  所有输入→相同输出? {'是 (退化!)' if all_same else '否 (有区分)'}")
