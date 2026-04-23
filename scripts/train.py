"""
NeuroFlow 训练脚本

模拟类脑学习过程：
1. 前向传播 → 显著性检测 → ECN/DMN 协作决策
2. 反向传播 + 记忆巩固（LTP 模拟）
3. 神经流形轨迹分析

用法:
    python scripts/train.py --epochs 50 --batch-size 32 --lr 0.001
"""

import argparse
import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflow.model import NeuroFlowModel


def generate_synthetic_data(
    n_samples: int = 1000,
    input_dim: int = 512,
    output_dim: int = 10,
    n_classes: int = 10,
):
    """
    生成合成数据集（模拟多模态神经信号）
    实际使用时替换为真实数据集加载逻辑
    """
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    # 添加可学习的模式
    for c in range(n_classes):
        mask = torch.arange(n_samples) % n_classes == c
        X[mask] += torch.randn(input_dim) * 0.5 + c * 0.3
    y = torch.arange(n_samples) % n_classes
    return X, y


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """单轮训练 — 包含记忆巩固步骤"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    ecn_gates = []
    dmn_gates = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # 前向传播
        result = model(batch_x, consolidate=(epoch % 5 == 0))
        output = result["output"]

        # 损失计算
        loss = criterion(output, batch_y)
        loss.backward()

        # 梯度裁剪（模拟突触可塑性上限）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

        # 记录门控状态（用于分析 ECN/DMN 切换）
        ecn_gates.append(result["ecn_gate"].mean().item())
        dmn_gates.append(result["dmn_gate"].mean().item())

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
        "avg_ecn_gate": sum(ecn_gates) / len(ecn_gates),
        "avg_dmn_gate": sum(dmn_gates) / len(dmn_gates),
    }


def evaluate(model, loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            result = model(batch_x)
            output = result["output"]
            loss = criterion(output, batch_y)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train NeuroFlow Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--memory-slots", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设备设置
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[NeuroFlow] Using device: {device}")

    torch.manual_seed(args.seed)

    # 数据集
    print("[NeuroFlow] Generating synthetic dataset...")
    X, y = generate_synthetic_data(
        n_samples=2000,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    )

    # 划分训练/验证集
    n_train = int(0.8 * len(X))
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 模型
    print("[NeuroFlow] Initializing model...")
    model = NeuroFlowModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[NeuroFlow] Model parameters: {n_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    os.makedirs(args.save_dir, exist_ok=True)
    history = []
    best_acc = 0

    print(f"\n[NeuroFlow] Starting training for {args.epochs} epochs...")
    print("-" * 60)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"ECN: {train_metrics['avg_ecn_gate']:.2f} DMN: {train_metrics['avg_dmn_gate']:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "ecn_gate": train_metrics["avg_ecn_gate"],
            "dmn_gate": train_metrics["avg_dmn_gate"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        # 保存最佳模型
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_acc,
                "config": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  ★ New best model saved (acc: {best_acc:.4f})")

    # 保存训练历史
    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"[NeuroFlow] Training complete in {time.time() - start_time:.1f}s")
    print(f"[NeuroFlow] Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
