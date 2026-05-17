"""NeuroFlow 推理能力全面测试"""
import neuroflow, numpy as np, time, platform, os

print("=" * 60)
print("  NeuroFlow 推理能力测试")
print("=" * 60)
print()

# 1. 基础信息
print("📋 模型信息")
print("-" * 40)
print(f"  版本:     v{neuroflow.__version__}")
print(f"  后端:     {neuroflow.get_backend()}")
print(f"  平台:     {platform.platform()}")

cpu_info = os.popen("lscpu | grep 'Model name' | cut -d: -f2 | xargs 2>/dev/null || echo Unknown").read().strip()
print(f"  CPU:      {cpu_info}")

from neuroflow import NeuroFlowLite, NeuroFlowModel, ModelConfig, benchmark

# 2. 模型参数
result = benchmark()
print()
print("📊 模型规模")
print("-" * 40)
print(f"  Full 模式: {result['original_params']:>10,} params  ({result['original_memory_mb']:.2f} MB)")
print(f"  Lite 模式: {result['lite_params']:>10,} params  ({result['lite_memory_mb']:.2f} MB)")
print(f"  压缩比:    {result['size_reduction']*100:>10.1f}%")

# 3. 推理速度测试
model = NeuroFlowLite(input_dim=512)
print()
print("⚡ 推理速度 (Lite C++ 模式)")
print("-" * 40)

def run_bench(name, batch, dim, iters=100):
    x = np.random.randn(batch, dim).astype(np.float32)
    for _ in range(5):
        _ = model.forward(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.forward(x)
    t = (time.perf_counter() - t0) / iters * 1000
    throughput = iters / (time.perf_counter() - t0)
    print(f"  {name:<20s}  batch={batch:>4d}  dim={dim:>4d}  {t:>8.3f}ms  ({throughput:>8.0f} samples/s)")
    return t

run_bench("单样本推理", 1, 512)
run_bench("小批量推理", 8, 512)
run_bench("中批量推理", 32, 512)
run_bench("大批量推理", 128, 512)
run_bench("高维输入", 1, 1024)

# 4. 输出结构
print()
print("🔍 输出 Tensor 结构 (batch=1, dim=512)")
print("-" * 40)
x = np.random.randn(1, 512).astype(np.float32)
output = model.forward(x)

fields = [
    "output", "decision", "value", "saliency",
    "ecn_gate", "dmn_gate", "anomaly",
    "mem_attention", "retrieved_mem",
]
for f in fields:
    try:
        val = getattr(output, f)
        if val is not None and hasattr(val, "shape"):
            s = val.shape
            mn, mx = val.min(), val.max()
            print(f"  {f:<20s}  shape={str(s):<15s}  range=[{mn:+.4f}, {mx:+.4f}]")
    except Exception:
        pass

# 5. 统计
stats = model.get_stats()
print()
print("📈 模型统计")
print("-" * 40)
print(f"  总参数:    {stats.total_params:>10,}")
print(f"  内存占用:  {stats.memory_bytes/1024:>10.1f} KB")
print(f"  量化比:    {stats.quantization_ratio*100:>10.1f}%")

# 6. 连续推理稳定性
print()
print("🔄 连续推理稳定性 (1000次)")
print("-" * 40)
times = []
x = np.random.randn(1, 512).astype(np.float32)
for _ in range(1000):
    t0 = time.perf_counter()
    _ = model.forward(x)
    times.append((time.perf_counter() - t0) * 1000)
times = np.array(times)
print(f"  均值:  {times.mean():.3f}ms")
print(f"  标准差: {times.std():.3f}ms")
print(f"  最小:  {times.min():.3f}ms")
print(f"  最大:  {times.max():.3f}ms")
print(f"  P99:   {np.percentile(times, 99):.3f}ms")

print()
print("=" * 60)
print("  ✅ 推理测试全部完成")
print("=" * 60)
