"""
NeuroFlow 推理脚本

用法:
    python scripts/inference.py --checkpoint checkpoints/best_model.pt --input-dim 512
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflow.model import NeuroFlowModel


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    model = NeuroFlowModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        memory_slots=config["memory_slots"],
        memory_dim=config["memory_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def analyze_saliency(model, x):
    """分析模型对输入的显著性检测"""
    result = model(x)
    return {
        "saliency": result["saliency"].item(),
        "ecn_gate": result["ecn_gate"].item(),
        "dmn_gate": result["dmn_gate"].item(),
        "anomaly": result["anomaly"].item(),
        "value": result["value"].item(),
        "prediction": result["output"].argmax(-1).item(),
        "confidence": torch.softmax(result["output"], dim=-1).max().item(),
    }


@torch.no_grad()
def trace_manifold(model, x, steps=20):
    """追踪神经流形轨迹"""
    trajectory = model.get_manifold_trajectory(x, steps=steps)
    return trajectory.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[NeuroFlow] Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"[NeuroFlow] Model loaded (val_acc: {checkpoint['val_acc']:.4f})")

    # 测试推理
    print("\n[NeuroFlow] Running inference on random sample...")
    test_input = torch.randn(1, args.input_dim).to(device)
    analysis = analyze_saliency(model, test_input)

    print(f"  Prediction:      {analysis['prediction']}")
    print(f"  Confidence:      {analysis['confidence']:.4f}")
    print(f"  Saliency:        {analysis['saliency']:.4f}")
    print(f"  ECN Gate:        {analysis['ecn_gate']:.4f}")
    print(f"  DMN Gate:        {analysis['dmn_gate']:.4f}")
    print(f"  Anomaly Score:   {analysis['anomaly']:.4f}")
    print(f"  Value Estimate:  {analysis['value']:.4f}")

    # 流形轨迹
    print(f"\n[NeuroFlow] Computing manifold trajectory ({args.steps} steps)...")
    trajectory = trace_manifold(model, test_input, steps=args.steps)
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  [NeuroFlow] Inference complete.")


if __name__ == "__main__":
    main()
