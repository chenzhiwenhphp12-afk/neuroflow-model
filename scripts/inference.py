"""
NeuroFlow 推理脚本 — 支持真实数据集推理 + 网络动态分析

用法:
    python scripts/inference.py --checkpoint neuroflow_checkpoint.pt --dataset digits --n-samples 5
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflow.model import NeuroFlowModel


def load_model(checkpoint_path, device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # 从模型权重推断 input_dim
    state_dict = checkpoint["model_state_dict"]
    input_proj_weight = state_dict.get("input_proj.0.weight")
    if input_proj_weight is not None:
        input_dim = input_proj_weight.shape[1]
    else:
        input_dim = config.get("input_dim", 512)

    n_classes = checkpoint.get("n_classes", config.get("output_dim", 10))

    model_config = {
        "input_dim": input_dim,
        "output_dim": n_classes,
    }
    model_config.update(config)
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = n_classes

    model = NeuroFlowModel(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def print_ascii_image(tensor, width=28, height=28, chars=" .:-=+*#%@"):
    """将图像张量打印为 ASCII 艺术"""
    img = tensor.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    scaled = (img * (len(chars) - 1)).astype(int)

    print("  ┌" + "─" * width + "┐")
    for row in scaled:
        line = "".join(chars[min(val, len(chars) - 1)] for val in row)
        print(f"  │{line}│")
    print("  └" + "─" * width + "┘")


def analyze_single_sample(model, x, class_names=None):
    """分析单个样本的推理过程"""
    with torch.no_grad():
        result = model(x)
        logits = result["decision"]
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(-1).item()
        confidence = probs.max().item()

    print(f"\n  ┌─ 推理结果 ──────────────────────┐")
    print(f"  │  预测类别: {pred}")
    if class_names:
        print(f"  │  类别名称: {class_names[pred]}")
    print(f"  │  置信度:   {confidence:.4f}")
    print(f"  │  显著性:   {result.get('saliency', torch.tensor(0)).item():.4f}")
    print(f"  │  ECN 门控: {result.get('ecn_gate', torch.tensor(0)).item():.4f}")
    print(f"  │  DMN 门控: {result.get('dmn_gate', torch.tensor(0)).item():.4f}")
    print(f"  │  异常评分: {result.get('anomaly', torch.tensor(0)).item():.4f}")
    print(f"  │  价值评估: {result.get('value', torch.tensor(0)).item():.4f}")
    print(f"  └──────────────────────────────────┘")

    # 显示 Top-5 预测
    top5 = probs.squeeze().topk(min(5, len(probs.squeeze())))
    print(f"\n  Top-5 预测:")
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        bar = "█" * int(prob.item() * 30)
        name = class_names[idx.item()] if class_names else str(idx.item())
        print(f"    {i+1}. {name:>12s} {prob:.4f} {bar}")


@torch.no_grad()
def trace_manifold(model, x, steps=20):
    """追踪神经流形轨迹"""
    trajectory = model.get_manifold_trajectory(x, steps=steps)
    return trajectory.squeeze(0).cpu().numpy()


def print_manifold_summary(trajectory):
    """打印流形轨迹摘要"""
    print(f"\n  ┌─ 神经流形轨迹 ──────────────────┐")
    print(f"  │  轨迹维度: {trajectory.shape}")
    print(f"  │  起点: [{trajectory[0, 0]:.4f}, {trajectory[0, 1]:.4f}, {trajectory[0, 2]:.4f}]")
    print(f"  │  终点: [{trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f}, {trajectory[-1, 2]:.4f}]")
    dist = ((trajectory[0] - trajectory[-1]) ** 2).sum() ** 0.5
    print(f"  │  总位移: {dist:.4f}")
    print(f"  └──────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="digits",
                        choices=["synthetic", "digits"])
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples to analyze")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or "cpu")

    print(f"[NeuroFlow] Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"[NeuroFlow] Model loaded (val_acc: {checkpoint['val_acc']:.4f})")

    input_dim = checkpoint.get("input_dim", 512)
    class_names = [str(i) for i in range(10)]

    if args.dataset == "digits":
        from sklearn.datasets import load_digits
        from sklearn.preprocessing import StandardScaler

        digits = load_digits()
        X = torch.tensor(digits.data[:args.n_samples], dtype=torch.float32)
        y = torch.tensor(digits.target[:args.n_samples], dtype=torch.long)

        # 投影到 784 维
        projection = torch.randn(64, 784) * 0.1
        X = X @ projection

        scaler = StandardScaler()
        X_np = scaler.fit_transform(X.numpy())
        X = torch.tensor(X_np, dtype=torch.float32)

        print(f"[NeuroFlow] Analyzing {args.n_samples} digits samples...")

        for i in range(args.n_samples):
            x = X[i:i+1].to(device)
            label = y[i].item()

            print(f"\n  样本 #{i} (真实标签: {label})")
            analyze_single_sample(model, x, class_names)
            trajectory = trace_manifold(model, x, steps=args.steps)
            print_manifold_summary(trajectory)

    elif args.dataset == "synthetic":
        print(f"[NeuroFlow] Analyzing {args.n_samples} synthetic samples...")
        for i in range(args.n_samples):
            x = torch.randn(1, input_dim).to(device)
            analyze_single_sample(model, x)
            trajectory = trace_manifold(model, x, steps=args.steps)
            print_manifold_summary(trajectory)

    print(f"\n[NeuroFlow] Inference complete.")


if __name__ == "__main__":
    main()
