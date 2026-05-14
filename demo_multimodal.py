"""
NeuroFlow 多模态 Demo — 视觉 + 文本 + 音频
===========================================
演示：
1. 视觉理解 — 生成真实图像（圆形/方形/条纹），NeuroFlow 提取视觉特征
2. 文本理解 — 文字描述输入，脑网络推理决策
3. 音频处理 — FFT 频谱特征提取 + NeuroFlow 处理
4. 跨模态对齐 — 文本-图像相似度，判断「描述是否匹配图像」
5. 显著性检测 — SN 网络自动关注关键区域

架构：
    图像 [3,224,224] ──► VisionEncoder ──┐
                                          ├── CrossModalFusion ──► Brain Networks
    文本 [batch,512]  ──► TextProject ────┘         │
                                          SN → ECN → DMN → Memory
    音频 [batch,256]  ──► AudioProject ──────────────┘
"""

import sys, numpy as np, time
sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import create_multimodal

# ============================================================
# 图像生成器 — 创建真实视觉刺激
# ============================================================
def make_image(pattern="circle", size=224):
    """生成测试图像：圆形、方形、条纹、渐变"""
    img = np.zeros((3, size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    if pattern == "circle":
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        img[0] = (dist < size // 3).astype(np.float32)  # R channel
        img[1] = (dist < size // 4).astype(np.float32)  # G channel
        img[2] = (dist < size // 6).astype(np.float32)  # B channel

    elif pattern == "square":
        s = size // 4
        img[0, cy - s : cy + s, cx - s : cx + s] = 1.0
        img[1, cy - s // 2 : cy + s // 2, cx - s // 2 : cx + s // 2] = 1.0

    elif pattern == "stripes":
        for i in range(0, size, 16):
            img[0, i : i + 8, :] = 1.0
        for i in range(8, size, 16):
            img[1, i : i + 8, :] = 1.0

    elif pattern == "gradient":
        X = np.linspace(0, 1, size).reshape(1, -1)
        img[0] = X
        img[1] = 1 - X
        img[2] = np.abs(np.sin(X * np.pi * 4))

    elif pattern == "triangle":
        for y in range(size):
            w = int((y / size) * size)
            x0 = max(0, cx - w // 2)
            x1 = min(size, cx + w // 2)
            img[0, y, x0:x1] = 1.0
            img[2, y, x0:x1] = 0.5

    elif pattern == "noise":
        img = np.random.randn(3, size, size).astype(np.float32) * 0.3

    return np.clip(img, 0, 1)


def text_to_features(text, dim=512):
    """将文本转为特征向量（简单哈希编码）"""
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text):
        idx = (ord(ch) * 7 + i * 13) % dim
        vec[idx] += 1.0 / max(len(text), 1)
    return vec


def audio_to_features(signal, dim=256, sample_rate=16000):
    """
    音频 → 频谱特征。
    signal: 1D numpy array of audio samples
    返回: Mel-like 频谱特征向量
    """
    # STFT-like: frame + FFT
    frame_size = 256
    hop = 128
    n_frames = (len(signal) - frame_size) // hop + 1
    if n_frames < 1:
        return np.zeros(dim, dtype=np.float32)

    n_frames = min(n_frames, 100)
    spectrum = np.zeros((n_frames, frame_size // 2), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + frame_size]
        if len(frame) < frame_size:
            break
        fft = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
        spectrum[i] = fft[: frame_size // 2]

    # Mel-scale compression: average over frequency bins → dim
    spec_mean = spectrum.mean(axis=0)
    repeats = dim // len(spec_mean) + 1
    features = np.tile(spec_mean, repeats)[:dim].astype(np.float32)
    return features / (features.max() + 1e-8)


# ============================================================
# 主 Demo
# ============================================================
print("=" * 64)
print("  🧠 NeuroFlow 多模态 Demo")
print("  视觉 + 文本 + 音频 → 类脑理解")
print("=" * 64)

# 初始化多模态模型
print("\n📋 初始化 NeuroFlowMultiModal...")
mm = create_multimodal(text_dim=512, image_size=224, output_dim=10)
print("  ✓ 模型就绪")
print(f"    Vision: ViT-style (224×224, 16×16 patches, 4 layers)")
print(f"    Fusion: Cross-Modal Text↔Image Alignment")
print(f"    Brain:  SN → ECN → DMN → Memory")

# ============================================================
# 实验 1: 视觉理解
# ============================================================
print("\n" + "=" * 64)
print("  👁️ 实验 1: 视觉理解 — 不同形状的视觉特征")
print("=" * 64)

patterns = {
    "circle":  "圆形 — 太阳/眼球/细胞",
    "square":  "方形 — 窗户/盒子/建筑",
    "stripes": "条纹 — 斑马/窗帘/纹理",
    "gradient":"渐变 — 天空/日落/深度",
    "triangle":"三角形 — 山/箭头/金字塔",
}

text_base = "a bright colorful shape"

for pattern, desc in patterns.items():
    img = make_image(pattern).reshape(1, 3, 224, 224).astype(np.float32)
    txt = text_to_features(text_base).reshape(1, -1)

    out = mm.forward_multimodal(txt, img)

    sim = out.text_image_sim[0, 0]
    sal = out.saliency[0, 0] if out.saliency.size > 0 else 0
    dec = out.decision[0]

    print(f"  {pattern:<10s} {desc:<30s} | sim={sim:+.3f} sal={sal:+.3f} "
          f"dec={dec[:3].round(2)}")


# ============================================================
# 实验 2: 跨模态对齐 — 文本描述 vs 图像匹配
# ============================================================
print("\n" + "=" * 64)
print("  🔗 实验 2: 跨模态对齐 — 文本描述是否匹配图像？")
print("=" * 64)

test_cases = [
    ("circle",  "a round red circle", "expected: high similarity"),
    ("circle",  "a blue square box",  "expected: low similarity"),
    ("square",  "a box with corners", "expected: high similarity"),
    ("square",  "a curved circle",    "expected: low similarity"),
    ("stripes", "horizontal lines",   "expected: high similarity"),
    ("stripes", "a solid red ball",   "expected: low similarity"),
    ("triangle","a mountain shape",   "expected: moderate"),
]

for pattern, text_desc, expected in test_cases:
    img = make_image(pattern).reshape(1, 3, 224, 224).astype(np.float32)
    txt = text_to_features(text_desc).reshape(1, -1)

    out = mm.forward_multimodal(txt, img)
    sim = float(out.text_image_sim[0, 0])

    # Normalize to [0, 1] for readability
    sim_norm = 1 / (1 + np.exp(-sim * 5))  # sigmoid
    bar = "█" * int(sim_norm * 20) + "░" * (20 - int(sim_norm * 20))

    print(f"  [{pattern:8s}] '{text_desc:30s}' → sim={sim:+.3f} |{bar}| {expected}")


# ============================================================
# 实验 3: 音频处理
# ============================================================
print("\n" + "=" * 64)
print("  🎵 实验 3: 音频处理 — 不同频率的神经响应")
print("=" * 64)

# 生成不同频率的音频信号
for freq, label in [(440, "A4 440Hz"), (880, "A5 880Hz"), (220, "A3 220Hz"), (1000, "C6 ~1kHz")]:
    t = np.linspace(0, 0.5, 8000, dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t) * 0.5

    audio_vec = audio_to_features(signal, dim=512)
    audio_input = audio_vec.reshape(1, -1)

    # Process through the multimodal model using text path (reuse for audio)
    out = mm.forward_text(audio_input)

    sal = float(out.saliency[0, 0])
    dec = out.decision[0]
    print(f"  {label:<15s} freq={freq:4d}Hz | saliency={sal:+.3f} "
          f"decision={dec[:3].round(3)}")


# ============================================================
# 实验 4: 显著性检测 — 不同刺激的注意力
# ============================================================
print("\n" + "=" * 64)
print("  🎯 实验 4: 显著性检测 — SN 网络对不同刺激的反应")
print("=" * 64)

stimuli = [
    ("bright circle",  make_image("circle"),    "视觉: 圆形"),
    ("dark square",    make_image("square"),     "视觉: 方形"),
    ("random noise",   make_image("noise"),      "视觉: 噪声"),
    ("calm text",      text_to_features("peace and harmony"), "文本: 平静"),
    ("urgent text",    text_to_features("DANGER WARNING ALERT URGENT"), "文本: 紧急"),
    ("440Hz tone",     audio_to_features(np.sin(2*np.pi*440*np.linspace(0,0.5,8000))*0.5, dim=512),
     "音频: 纯音"),
]

for name, features, modality in stimuli:
    if features.ndim == 1:
        x = features.reshape(1, -1)
    elif features.ndim == 3:
        x = features.reshape(1, 3, 224, 224)
    else:
        x = features.reshape(1, -1)

    if modality.startswith("视觉"):
        txt = text_to_features("a shape").reshape(1, -1)
        out = mm.forward_multimodal(txt, x)
    else:
        out = mm.forward_text(x)

    sal = float(out.saliency[0, 0]) if out.saliency.ndim >= 2 else float(out.saliency)
    anom_val = out.anomaly
    anomaly = float(anom_val[0, 0]) if anom_val.ndim >= 2 else (float(anom_val) if anom_val.size > 0 else 0)
    bar = "█" * int(abs(sal) * 30) + ("⚠" if anomaly > 0.5 else "✓")

    print(f"  {name:<20s} {modality:<15s} "
          f"sal={sal:+.3f} anomaly={anomaly:+.3f} |{bar}")


# ============================================================
# 性能统计
# ============================================================
print("\n" + "=" * 64)
print("  ⚡ 性能统计")
print("=" * 64)

img = make_image("circle").reshape(1, 3, 224, 224).astype(np.float32)
txt = text_to_features("a red circle").reshape(1, -1)

# Warmup
for _ in range(5):
    _ = mm.forward_multimodal(txt, img)

# Benchmark
t0 = time.perf_counter()
for _ in range(50):
    _ = mm.forward_multimodal(txt, img)
elapsed = (time.perf_counter() - t0) / 50 * 1000

print(f"  多模态推理 (文本+图像): {elapsed:.1f}ms")
print(f"  吞吐量: {1000/elapsed:.0f} samples/s")
print(f"  模型: Vision(ViT 4层) + Fusion + SN/ECN/DMN + Memory")
print(f"  输出: decision + value + saliency + text_image_sim + anomaly")

print("\n" + "=" * 64)
print("  ✅ 多模态 Demo 完成")
print("  🧠 视觉 ✓  文本 ✓  音频 ✓  跨模态对齐 ✓")
print("=" * 64)
