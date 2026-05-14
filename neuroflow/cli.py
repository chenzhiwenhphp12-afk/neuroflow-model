"""
NeuroFlow CLI - command-line interface for benchmarking and inference.
"""

import time
import argparse
import numpy as np


def benchmark():
    """Run a quick benchmark of the NeuroFlow model."""
    parser = argparse.ArgumentParser(description="NeuroFlow Model Benchmark")
    parser.add_argument("--mode", choices=["lite", "full"], default="lite",
                        help="Model variant (default: lite)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of forward passes (default: 1000)")
    parser.add_argument("--input-dim", type=int, default=512,
                        help="Input dimension (default: 512)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NeuroFlow Model Benchmark")
    print(f"  Mode: {args.mode.upper()} | Dim: {args.input_dim} | Iters: {args.iterations}")
    print(f"{'='*60}\n")

    try:
        from neuroflow import NeuroFlowModel, NeuroFlowLite
        use_cpp = True
    except ImportError:
        print("[!] C++ core not available, using Python fallback")
        use_cpp = False

    if use_cpp:
        if args.mode == "lite":
            model = NeuroFlowLite(input_dim=args.input_dim)
        else:
            from neuroflow import ModelConfig
            cfg = ModelConfig()
            cfg.input_dim = args.input_dim
            cfg.hidden_dim = 256
            cfg.output_dim = 10
            cfg.use_quantization = (args.mode == "lite")
            model = NeuroFlowModel(cfg)
    else:
        from neuroflow.model_lite import NeuroFlowModelLite
        model = NeuroFlowModelLite(input_dim=args.input_dim, output_dim=10)

    x = np.random.randn(1, args.input_dim).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = model.forward(x)

    # Benchmark
    start = time.perf_counter()
    for _ in range(args.iterations):
        _ = model.forward(x)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / args.iterations) * 1000
    throughput = args.iterations / elapsed

    if use_cpp:
        stats = model.get_stats()
        params = stats.total_params
        mem_kb = stats.memory_bytes / 1024
    else:
        params = model.count_parameters()
        mem_kb = 0

    print(f"  Results:")
    print(f"    Total time:      {elapsed:.3f}s")
    print(f"    Avg per forward: {avg_ms:.3f}ms")
    print(f"    Throughput:      {throughput:.0f} samples/s")
    print(f"    Parameters:      {params:,}")
    print(f"    Memory:          {mem_kb:.1f} KB")
    print(f"    Backend:         {'C++ (SIMD)' if use_cpp else 'Python'}")
    print()


if __name__ == "__main__":
    benchmark()
