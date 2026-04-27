"""
NeuroFlow V2 训练脚本

支持:
- 知识蒸馏 (从大模型或预计算 logits)
- 动态量化训练
- 效率监控
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_v2 import NeuroFlowV2


class IndexedTensorDataset(TensorDataset):
    """带索引的数据集"""
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)


def load_digits_dataset():
    """sklearn digits 数据集"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    
    # 投影到目标维度
    projection = torch.randn(64, 512) * 0.1
    X = X @ projection
    
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X.numpy())
    X = torch.tensor(X_np, dtype=torch.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_val, y_val), 512, 10


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    teacher_fn=None,
    temperature=2.0,
    alpha=0.5,
    use_cache=True,
):
    """
    训练一个 epoch
    
    Args:
        model: NeuroFlowV2 模型
        teacher_fn: 教师模型函数 (可选)
        temperature: 蒸馏温度
        alpha: Hard Loss 权重
        use_cache: 是否启用推理缓存
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    routing_stats = []
    
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    for data in dataloader:
        if len(data) == 3:
            batch_x, batch_y, batch_indices = data
        else:
            batch_x, batch_y = data
            batch_indices = None
        
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # V2 推理
        result = model(batch_x, use_cache=use_cache)
        logits = result["output"]
        
        # Hard Loss
        loss_hard = criterion(logits, batch_y)
        loss = loss_hard
        
        # Soft Loss (蒸馏)
        if teacher_fn is not None and alpha < 1.0:
            with torch.no_grad():
                if batch_indices is not None:
                    teacher_logits = teacher_fn(batch_x, indices=batch_indices)
                else:
                    teacher_logits = teacher_fn(batch_x)
            
            student_soft = F.log_softmax(logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits.to(device) / temperature, dim=1)
            
            loss_kd = kd_loss_fn(student_soft, teacher_soft) * (temperature ** 2)
            loss = alpha * loss_hard + (1 - alpha) * loss_kd
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = logits.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
        # 收集路由统计
        if "routing_info" in result:
            routing_stats.append(result["routing_info"].get("activated_ratio", 0))
    
    avg_loss = total_loss / total
    accuracy = correct / total
    avg_activation = sum(routing_stats) / len(routing_stats) if routing_stats else 0
    
    return avg_loss, accuracy, avg_activation


def validate(model, dataloader, criterion, device, use_cache=True):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    cache_hits = 0
    cache_total = 0
    
    with torch.no_grad():
        for data in dataloader:
            if len(data) == 3:
                batch_x, batch_y, _ = data
            else:
                batch_x, batch_y = data
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            result = model(batch_x, use_cache=use_cache)
            logits = result["output"]
            
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            if result.get("cache_hit", False):
                cache_hits += 1
            cache_total += 1
    
    avg_loss = total_loss / total
    accuracy = correct / total
    cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0
    
    return avg_loss, accuracy, cache_hit_rate


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow V2 训练")
    parser.add_argument("--dataset", type=str, default="digits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--activated-experts", type=int, default=1)
    parser.add_argument("--save", type=str, default="neuroflow_v2_checkpoint.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--teacher-logits", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true", help="训练后运行基准测试")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[NeuroFlow V2] Device: {device}")
    print(f"[NeuroFlow V2] 配置:")
    print(f"  hidden_dim: {args.hidden_dim}")
    print(f"  memory_slots: {args.memory_slots}")
    print(f"  num_experts: {args.num_experts}")
    print(f"  activated_experts: {args.activated_experts}")
    
    # 加载数据
    print(f"\n[NeuroFlow V2] Loading {args.dataset} dataset...")
    (X_train, y_train), (X_val, y_val), input_dim, n_classes = load_digits_dataset()
    
    train_dataset = IndexedTensorDataset(X_train, y_train)
    val_dataset = IndexedTensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[NeuroFlow V2] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 创建模型
    model = NeuroFlowV2(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=n_classes,
        memory_slots=args.memory_slots,
        num_experts=args.num_experts,
        activated_experts=args.activated_experts,
    ).to(device)
    
    eff = model.get_inference_efficiency()
    print(f"\n[NeuroFlow V2] Model initialized:")
    print(f"  总参数: {eff['params_total']:,}")
    print(f"  激活参数: {eff['params_activated']:,}")
    print(f"  激活比: {eff['activation_ratio']*100:.1f}%")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 教师模型设置
    teacher_fn = None
    if args.distill:
        if args.teacher_logits:
            # 预计算 logits
            print(f"\n[NeuroFlow V2] Loading precomputed teacher logits...")
            logits_data = torch.load(args.teacher_logits)
            teacher_fn = lambda x, indices=None: logits_data[indices] if indices is not None else logits_data[:x.size(0)]
        else:
            # 本地 MLP 教师
            print(f"\n[NeuroFlow V2] Training local MLP teacher...")
            
            class MLPTeacher(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, output_dim),
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            teacher = MLPTeacher(input_dim, 512, n_classes).to(device)
            t_opt = torch.optim.Adam(teacher.parameters(), lr=0.001)
            t_crit = nn.CrossEntropyLoss()
            
            X_all = torch.cat([X_train, X_val]).to(device)
            y_all = torch.cat([y_train, y_val]).to(device)
            
            for epoch in range(20):
                teacher.train()
                t_opt.zero_grad()
                out = teacher(X_all)
                loss = t_crit(out, y_all)
                loss.backward()
                t_opt.step()
            
            teacher.eval()
            teacher_fn = lambda x, indices=None: teacher(x)
    
    # 训练循环
    print(f"\n[NeuroFlow V2] Training for {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Acc':>7} | {'Val Acc':>8} | {'Activation':>10} | {'Time':>5}")
    print("-" * 60)
    
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_acc, avg_activation = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_fn=teacher_fn, temperature=args.temperature, alpha=args.alpha
        )
        
        val_loss, val_acc, cache_rate = validate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start
        
        print(f"{epoch:5d} | {train_loss:8.4f} | {train_acc:6.2%} | {val_acc:7.2%} | {avg_activation:10.2%} | {elapsed:4.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "output_dim": n_classes,
                    "memory_slots": args.memory_slots,
                    "num_experts": args.num_experts,
                },
                "efficiency": eff,
            }, args.save)
    
    print(f"\n[NeuroFlow V2] Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.2%}")
    print(f"  Checkpoint: {args.save}")
    
    # 基准测试
    if args.benchmark:
        print(f"\n[NeuroFlow V2] Running benchmark...")
        
        model.eval()
        x = torch.randn(32, input_dim).to(device)
        
        times = []
        for _ in range(20):
            start = time.time()
            with torch.no_grad():
                model(x)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000
        
        print(f"\n推理延迟: {avg_time:.2f} ms")
        print(f"缓存命中率: {model.inference_cache.hit_rate():.2%}")
        print(f"记忆历史: {len(model.memory_history)} events")


if __name__ == "__main__":
    main()