"""
NeuroFlow 训练脚本

使用方法:
    python train.py --config configs/train_config.yaml
    python train.py --task classification --epochs 50
    python train.py --task multimodal --epochs 100
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from neuroflow.model import NeuroFlowModel
from neuroflow.trainer import (
    NeuroFlowTrainer,
    MultiModalTrainer,
    create_synthetic_dataset,
    create_multimodal_dataset,
    SimpleDataset,
    MultiModalDataset,
)
from torch.utils.data import DataLoader, random_split


def parse_args():
    parser = argparse.ArgumentParser(description='NeuroFlow Training')
    
    # 任务类型
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'multimodal', 'online'],
                        help='训练任务类型')
    
    # 数据相关
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据路径（若为None则使用合成数据）')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='合成数据样本数')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--memory_dim', type=int, default=128)
    parser.add_argument('--memory_slots', type=int, default=64)
    
    # 多模态参数
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--text_dim', type=int, default=512)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    
    # 优化器
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'step', 'none'])
    
    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--export_cpp', type=bool, default=True)
    
    # 其他
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='YAML配置文件路径')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_classification(args):
    """训练分类任务"""
    print("=" * 60)
    print("NeuroFlow Classification Training")
    print("=" * 60)
    
    # 创建模型
    model = NeuroFlowModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        memory_dim=args.memory_dim,
        memory_slots=args.memory_slots,
    )
    
    # 创建数据
    if args.data_path:
        # 加载真实数据（用户自定义）
        print(f"Loading data from {args.data_path}")
        # TODO: 实现真实数据加载
        raise NotImplementedError("请实现真实数据加载逻辑")
    else:
        # 使用合成数据
        print(f"Creating synthetic dataset: {args.num_samples} samples")
        dataset = create_synthetic_dataset(
            num_samples=args.num_samples,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            memory_dim=args.memory_dim,
        )
    
    # 分割数据
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建训练器
    trainer = NeuroFlowTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=args.device,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
    )
    
    # 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        consolidate=True,
        save_dir=args.save_dir,
        save_best=args.save_best,
        log_interval=args.log_interval,
    )
    
    # 导出权重给C++
    if args.export_cpp:
        export_path = Path(args.save_dir) / 'neuroflow_weights.npz'
        trainer.export_weights_for_cpp(str(export_path))
    
    return history


def train_multimodal(args):
    """训练多模态任务"""
    print("=" * 60)
    print("NeuroFlow MultiModal Training")
    print("=" * 60)
    
    # 检查是否有多模态模型
    try:
        from neuroflow.multimodal_model import NeuroFlowMultiModal
    except ImportError:
        print("Warning: MultiModal model not available, using base model")
        return train_classification(args)
    
    # 创建多模态模型
    model = NeuroFlowMultiModal(
        text_dim=args.text_dim,
        image_size=args.image_size,
        output_dim=args.output_dim,
    )
    
    # 创建多模态数据
    print(f"Creating multimodal dataset: {args.num_samples} samples")
    dataset = create_multimodal_dataset(
        num_samples=args.num_samples,
        text_dim=args.text_dim,
        image_size=args.image_size,
        output_dim=args.output_dim,
    )
    
    # 分割数据
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建多模态训练器
    trainer = MultiModalTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=args.device,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
    )
    
    # 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        consolidate=True,
        save_dir=args.save_dir,
        save_best=args.save_best,
        log_interval=args.log_interval,
    )
    
    # 导出权重
    if args.export_cpp:
        export_path = Path(args.save_dir) / 'neuroflow_multimodal_weights.npz'
        trainer.export_weights_for_cpp(str(export_path))
    
    return history


def train_online(args):
    """在线学习演示"""
    print("=" * 60)
    print("NeuroFlow Online Learning Demo")
    print("=" * 60)
    
    # 创建模型
    model = NeuroFlowModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    )
    
    # 创建训练器
    trainer = NeuroFlowTrainer(
        model=model,
        lr=args.lr,
        device=args.device,
    )
    
    # 生成在线学习数据
    print("Testing online learning with few samples...")
    
    # 单样本快速适应
    x_single = torch.randn(1, args.input_dim)
    y_single = torch.tensor([3])  # 目标类别
    
    result = trainer.online_learn(x_single, y_single, num_steps=10)
    print(f"  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss reduction: {result['loss_reduction']:.4f}")
    
    # 少样本适应
    x_few = torch.randn(5, args.input_dim)
    y_few = torch.randint(0, args.output_dim, (5,))
    
    result = trainer.online_learn(x_few, y_few, num_steps=20)
    print(f"\nFew-shot adaptation (5 samples):")
    print(f"  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    
    # 记忆巩固演示
    print("\nMemory consolidation demo...")
    batch = torch.randn(32, args.input_dim)
    
    with torch.no_grad():
        # 获取记忆前
        mem_before = model.memory.memory_bank.data.clone()
        
        # 执行记忆巩固
        model(batch, consolidate=True)
        
        # 获取记忆后
        mem_after = model.memory.memory_bank.data.clone()
        
        # 计算变化
        change = (mem_after - mem_before).abs().mean().item()
        print(f"  Memory bank change: {change:.6f}")
        print(f"  (LTP learning rate: {model.memory.ltp_rate})")
    
    return result


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载配置文件（如果存在）
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # 根据任务类型训练
    if args.task == 'classification':
        history = train_classification(args)
    elif args.task == 'multimodal':
        history = train_multimodal(args)
    elif args.task == 'online':
        history = train_online(args)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()