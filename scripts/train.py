"""
NeuroFlow - 主训练脚本
支持多种数据集：synthetic, sklearn_digits, mnist_local
"""

import argparse
import json
import os
import sys
import time

# 确保可以导入 neuroflow 和 teacher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class IndexedTensorDataset(TensorDataset):
    """Returns (x, y, index) tuples"""
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)


def load_synthetic_dataset(input_dim=784, n_classes=10, n_samples=5000):
    """合成数据集，模拟皮层信号输入模式"""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    # 添加类间差异
    for c in range(n_classes):
        mask = y == c
        X[mask] += torch.randn(input_dim) * 0.5 + c * 0.3
    return X, y


def load_digits_dataset():
    """
    sklearn digits 数据集 (8x8 手写数字，64 维特征)
    无需联网下载，本地自带
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)

    # 上采样到 784 维以匹配 NeuroFlow 输入
    # 使用零填充 + 随机投影模拟从 64 到 784 的扩展
    n_samples, n_features = X.shape
    projection = torch.randn(n_features, 784) * 0.1
    X = X @ projection

    # 标准化
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X.numpy())
    X = torch.tensor(X_np, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_val, y_val)


def load_sklearn_dataset(name="wine"):
    """sklearn 内置数据集"""
    from sklearn.datasets import load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    datasets = {
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    loader = datasets.get(name, load_wine)
    data = loader()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)

    scaler = StandardScaler()
    X_np = scaler.fit_transform(X.numpy())
    X = torch.tensor(X_np, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_val, y_val)


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0,
                teacher_fn=None, temperature=2.0, alpha=0.5):
    """
    teacher_fn: 函数，接收 batch_x 返回 teacher_logits (batch, n_classes)
    temperature: 蒸馏温度
    alpha: 蒸馏权重 (0.0 = 无蒸馏, 1.0 = 纯蒸馏)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")

    for data in dataloader:
        if len(data) == 3:
            batch_x, batch_y, batch_indices = data
        else:
            batch_x, batch_y = data
            batch_indices = None
        
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        logits = output["decision"]

        # 1. Hard Loss (标准交叉熵)
        loss_hard = criterion(logits, batch_y)
        loss = loss_hard

        # 2. Soft Loss (蒸馏)
        if teacher_fn is not None and alpha < 1.0:
            with torch.no_grad():
                # 如果 teacher 支持 indices 参数 (PrecomputedTeacher)，传入它
                import inspect
                sig = inspect.signature(teacher_fn.__call__) if hasattr(teacher_fn, '__call__') else inspect.signature(teacher_fn)
                if 'indices' in sig.parameters:
                    teacher_logits = teacher_fn(batch_x, indices=batch_indices).to(device)
                else:
                    teacher_logits = teacher_fn(batch_x).to(device)
            
            # 软化分布
            student_soft = F.log_softmax(logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
            
            # KL 散度
            loss_kd = kd_loss_fn(student_soft, teacher_soft) * (temperature ** 2)
            
            # 混合损失
            loss = alpha * loss_hard + (1 - alpha) * loss_kd

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = logits.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            if len(data) == 3:
                batch_x, batch_y, _ = data
            else:
                batch_x, batch_y = data
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            logits = output["decision"]
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def analyze_network_dynamics(model, dataloader, device, n_batches=5):
    """分析网络动态：门控模式、能量景观"""
    model.eval()
    ecn_activations = []
    dmn_activations = []
    sn_gates = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= n_batches:
                break
            if len(data) == 3:
                batch_x, _, _ = data
            else:
                batch_x, _ = data
            batch_x = batch_x.to(device)
            output, dynamics = model(batch_x)

            ecn_activations.append(dynamics["ecn_output"].mean(dim=0))
            dmn_activations.append(dynamics["dmn_output"].mean(dim=0))
            sn_gates.append(dynamics["sn_gate"].mean(dim=0))

    ecn_mean = torch.stack(ecn_activations).mean(dim=0)
    dmn_mean = torch.stack(dmn_activations).mean(dim=0)
    sn_mean = torch.stack(sn_gates).mean(dim=0)

    print("\n[Network Dynamics Analysis]")
    print(f"  ECN activation mean: {ecn_mean.mean().item():.4f} (std: {ecn_mean.std().item():.4f})")
    print(f"  DMN activation mean: {dmn_mean.mean().item():.4f} (std: {dmn_mean.std().item():.4f})")
    print(f"  SN gate mean:        {sn_mean.mean().item():.4f} (range: [{sn_gates[0].min().item():.4f}, {sn_gates[0].max().item():.4f}])")


def print_ascii_image(img_2d):
    """ASCII 可视化"""
    chars = " .:-=+*#%@"
    if hasattr(img_2d, "numpy"):
        img_2d = img_2d.numpy()
    img_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
    img_ascii = np.char.array([chars[int(v * (len(chars) - 1))] for v in img_norm.flatten()])
    img_ascii = img_ascii.reshape(img_2d.shape)
    print("  " + "\n  ".join("".join(row) for row in img_ascii))


def analyze_single_sample(model, x, class_names=None):
    """单样本分析"""
    model.eval()
    with torch.no_grad():
        output, dynamics = model(x)

    probs = torch.softmax(output, dim=1)[0]
    _, predicted = output.max(1)

    top3_idx = probs.argsort(descending=True)[:3]
    print(f"  预测: {class_names[predicted.item()] if class_names else predicted.item()}")
    for idx in top3_idx:
        print(f"    {class_names[idx] if class_names else idx}: {probs[idx].item():.4f}")

    ecn_energy = dynamics["ecn_output"].pow(2).mean().item()
    dmn_energy = dynamics["dmn_output"].pow(2).mean().item()
    sn_gate = dynamics["sn_gate"].mean().item()
    print(f"  能量: ECN={ecn_energy:.4f}, DMN={dmn_energy:.4f}, SN={sn_gate:.4f}")


def trace_manifold(model, x, steps=20):
    """追踪流形轨迹"""
    model.eval()
    with torch.no_grad():
        _, dynamics = model(x)
        manifold = dynamics["manifold_projection"]

    trajectory = [manifold.clone()]
    for _ in range(steps):
        manifold = model.project_to_manifold(manifold)
        trajectory.append(manifold.clone())

    return trajectory


def print_manifold_summary(trajectory):
    """打印流形摘要"""
    print("  流形轨迹:")
    for i, m in enumerate(trajectory[::5]):
        print(f"    step {i*5}: mean={m.mean().item():.4f}, std={m.std().item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow 训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--dataset", type=str, default="digits",
                        choices=["synthetic", "digits", "wine", "breast_cancer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save", type=str, default="neuroflow_checkpoint.pt")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # 蒸馏参数
    parser.add_argument("--distill", action="store_true", help="启用知识蒸馏")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard Loss 权重 (0.0=纯蒸馏, 1.0=无蒸馏)")
    parser.add_argument("--teacher-epochs", type=int, default=20, help="教师模型预训练轮数 (仅 distill 模式)")
    parser.add_argument("--teacher-logits-file", type=str, default=None, help="预计算的教师软标签文件 (.pt)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NeuroFlow] Device: {device}")
    print(f"[NeuroFlow] Dataset: {args.dataset}")

    # 加载配置
    try:
        with open(args.config) as f:
            full_config = json.load(f)
        # 提取 model 配置
        config = full_config.get("model", full_config)
    except FileNotFoundError:
        print(f"[WARN] Config {args.config} not found, using defaults")
        config = {}

    # 加载数据集
    print(f"\n[NeuroFlow] Loading {args.dataset} dataset...")
    if args.dataset == "synthetic":
        X, y = load_synthetic_dataset()
        n_train = int(0.8 * len(X))
        train_dataset = IndexedTensorDataset(X[:n_train], y[:n_train])
        val_dataset = IndexedTensorDataset(X[n_train:], y[n_train:])
        input_dim = X.size(1)
        n_classes = int(y.max().item()) + 1

    elif args.dataset == "digits":
        (X_train, y_train), (X_val, y_val) = load_digits_dataset()
        input_dim = X_train.size(1)
        n_classes = 10
        train_dataset = IndexedTensorDataset(X_train, y_train)
        val_dataset = IndexedTensorDataset(X_val, y_val)

    elif args.dataset in ("wine", "breast_cancer"):
        (X_train, y_train), (X_val, y_val) = load_sklearn_dataset(args.dataset)
        input_dim = X_train.size(1)
        n_classes = int(y_train.max().item()) + 1
        train_dataset = IndexedTensorDataset(X_train, y_train)
        val_dataset = IndexedTensorDataset(X_val, y_val)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"[NeuroFlow] Input dim: {input_dim}, Classes: {n_classes}")
    print(f"[NeuroFlow] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    from neuroflow.model import NeuroFlowModel
    # config 先加载，然后被显式参数覆盖
    model_config = dict(config)
    model_config["input_dim"] = input_dim
    model_config["output_dim"] = n_classes
    model = NeuroFlowModel(**model_config).to(device)

    print(f"\n[NeuroFlow] Model initialized")
    print(f"[NeuroFlow] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # 蒸馏设置
    teacher_fn = None
    if args.distill:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teacher import MLPTeacher, TeacherWrapper, PrecomputedTeacher, train_teacher
        
        if args.teacher_logits_file:
            # 模式 1: 使用预计算的软标签 (适合 LLM)
            print(f"\n[NeuroFlow] Loading Precomputed Teacher Logits from {args.teacher_logits_file}...")
            teacher_fn = PrecomputedTeacher(args.teacher_logits_file)
            print(f"[NeuroFlow] Precomputed Teacher Ready. Shape: {teacher_fn.logits.shape}")
        else:
            # 模式 2: 本地 MLP 教师
            print(f"\n[NeuroFlow] Preparing Local MLP Teacher Model for Distillation...")
            teacher = MLPTeacher(
                input_dim=input_dim, 
                hidden_dim=512,
                output_dim=n_classes
            ).to(device)
            
            # 训练教师
            X_all = torch.cat([train_dataset.tensors[0], val_dataset.tensors[0]])
            y_all = torch.cat([train_dataset.tensors[1], val_dataset.tensors[1]])
            train_teacher(teacher, X_all, y_all, epochs=args.teacher_epochs, lr=0.001)
            
            teacher_fn = TeacherWrapper(teacher, device)
            print(f"[NeuroFlow] MLP Teacher Ready. Distillation: alpha={args.alpha}, T={args.temperature}")

    # 训练循环
    print(f"\n[NeuroFlow] Starting training for {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>8} | {'Val Loss':>10} | {'Val Acc':>8} | {'Time':>6}")
    print("-" * 65)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip,
            teacher_fn=teacher_fn, temperature=args.temperature, alpha=args.alpha
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:7.2%} | {val_loss:10.4f} | {val_acc:7.2%} | {elapsed:5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config,
            }, args.save)

    print(f"\n[NeuroFlow] Training complete!")
    print(f"[NeuroFlow] Best validation accuracy: {best_val_acc:.2%}")
    print(f"[NeuroFlow] Checkpoint saved to {args.save}")

    # 网络动态分析
    if args.analyze:
        print("\n[NeuroFlow] Analyzing network dynamics...")
        analyze_network_dynamics(model, val_loader, device)


if __name__ == "__main__":
    main()
