"""
NeuroFlow 训练器

支持：
- 多任务训练（分类、多模态）
- 多种损失函数
- 多种优化器
- 学习率调度
- 模型保存/加载
- 权重导出到C++
- 在线学习（LTP记忆巩固）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, List, Tuple, Any, Callable
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import math


class NeuroFlowTrainer:
    """
    NeuroFlow 模型训练器
    
    Args:
        model: NeuroFlow模型实例
        lr: 学习率
        weight_decay: 权重衰减
        optimizer: 优化器类型 ('adam', 'adamw', 'sgd', 'rmsprop')
        scheduler: 学习率调度器 ('cosine', 'linear', 'step', 'none')
        device: 训练设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer: str = 'adamw',
        scheduler: str = 'cosine',
        device: str = 'auto',
        grad_clip: float = 1.0,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer(optimizer)
        
        # 调度器（在train中设置num_epochs后初始化）
        self.scheduler_type = scheduler
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
        }
    
    def _create_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """创建优化器"""
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        if optimizer_type == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=self.lr)
        elif optimizer_type == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        elif optimizer_type == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=self.lr, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(optimizer_grouped_parameters, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self, num_epochs: int, steps_per_epoch: int):
        """创建学习率调度器"""
        total_steps = num_epochs * steps_per_epoch
        
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.lr * 0.01
            )
        elif self.scheduler_type == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=total_steps // 3, gamma=0.1
            )
        else:
            self.scheduler = None
    
    def _get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def _warmup_lr(self, step: int):
        """学习率预热"""
        if step < self.warmup_steps:
            warmup_lr = self.lr * (step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        consolidate: bool = True,
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 学习率预热
            if self.global_step < self.warmup_steps:
                self._warmup_lr(self.global_step)
            
            # 获取数据
            if isinstance(batch, dict):
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                memory_input = batch.get('memory_input', None)
                if memory_input is not None:
                    memory_input = memory_input.to(self.device)
            else:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                memory_input = None
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(
                x, 
                memory_input=memory_input,
                consolidate=consolidate,
                return_manifold=False,
            )
            
            # 计算损失
            loss = loss_fn(outputs['output'], y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # 参数更新
            self.optimizer.step()
            
            # 调度器更新
            if self.scheduler is not None and self.global_step >= self.warmup_steps:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item() * x.size(0)
            if outputs['output'].dim() > 1 and outputs['output'].size(1) > 1:
                preds = outputs['output'].argmax(dim=-1)
                if y.dim() > 1 and y.size(1) > 1:
                    y = y.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch['input'].to(self.device)
                y = batch['target'].to(self.device)
                memory_input = batch.get('memory_input', None)
                if memory_input is not None:
                    memory_input = memory_input.to(self.device)
            else:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                memory_input = None
            
            outputs = self.model(
                x,
                memory_input=memory_input,
                consolidate=False,
                return_manifold=False,
            )
            
            loss = loss_fn(outputs['output'], y)
            
            total_loss += loss.item() * x.size(0)
            if outputs['output'].dim() > 1 and outputs['output'].size(1) > 1:
                preds = outputs['output'].argmax(dim=-1)
                if y.dim() > 1 and y.size(1) > 1:
                    y = y.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        loss_fn: Optional[Callable] = None,
        consolidate: bool = True,
        save_dir: Optional[str] = None,
        save_best: bool = True,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            loss_fn: 损失函数（默认CrossEntropyLoss）
            consolidate: 是否启用记忆巩固
            save_dir: 模型保存目录
            save_best: 是否只保存最佳模型
            log_interval: 日志打印间隔
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # 创建调度器
        self._create_scheduler(num_epochs, len(train_loader))
        
        # 保存目录
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f"Validation samples: {len(val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader, loss_fn, consolidate)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['lr'].append(self._get_lr())
            
            # 验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, loss_fn)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # 日志
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{num_epochs} | "
                msg += f"Train Loss: {train_metrics['loss']:.4f} | "
                msg += f"Train Acc: {train_metrics['accuracy']:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_metrics['loss']:.4f} | "
                    msg += f"Val Acc: {val_metrics['accuracy']:.4f}"
                msg += f" | LR: {self._get_lr():.6f}"
                print(msg)
            
            # 保存模型
            if save_dir is not None:
                current_loss = val_metrics['loss'] if val_loader is not None else train_metrics['loss']
                
                if not save_best or current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_checkpoint(save_dir / 'best_model.pt', epoch, current_loss)
                    print(f"  -> Saved best model (loss: {current_loss:.4f})")
                
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch+1}.pt', epoch, current_loss)
        
        print("-" * 50)
        print(f"Training completed. Best loss: {self.best_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}, loss: {checkpoint['loss']:.4f}")
    
    def export_weights_for_cpp(self, output_path: str):
        """
        导出权重为C++可读格式
        
        Args:
            output_path: 输出文件路径（.npz格式）
        """
        weights = {}
        
        for name, param in self.model.named_parameters():
            # 转换为numpy并保存
            key = name.replace('.', '_')
            weights[key] = param.data.cpu().numpy()
            print(f"  {key}: {param.shape}")
        
        # 保存为npz
        np.savez(output_path, **weights)
        print(f"\nWeights exported to: {output_path}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 同时保存元数据
        metadata = {
            'model_type': self.model.__class__.__name__,
            'parameters': {name: list(param.shape) for name, param in self.model.named_parameters()},
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'export_time': datetime.now().isoformat(),
        }
        
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return weights
    
    def online_learn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 5,
        lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        在线学习（单样本或少样本快速适应）
        
        Args:
            x: 输入数据 (batch, input_dim)
            y: 目标 (batch,) 或 (batch, output_dim)
            num_steps: 适应步数
            lr: 学习率（默认使用训练器学习率的10倍）
        """
        if lr is None:
            lr = self.lr * 10
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss() if y.dim() == 1 else nn.MSELoss()
        
        x, y = x.to(self.device), y.to(self.device)
        
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            outputs = self.model(x, consolidate=True)
            loss = loss_fn(outputs['output'], y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return {
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'loss_reduction': losses[0] - losses[-1],
            'losses': losses,
        }


class MultiModalTrainer(NeuroFlowTrainer):
    """多模态训练器"""
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        consolidate: bool = True,
    ) -> Dict[str, float]:
        """训练一个epoch（多模态版本）"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if self.global_step < self.warmup_steps:
                self._warmup_lr(self.global_step)
            
            # 多模态数据
            if isinstance(batch, dict):
                text = batch.get('text', None)
                image = batch.get('image', None)
                y = batch['target'].to(self.device)
                
                if text is not None:
                    text = text.to(self.device)
                if image is not None:
                    image = image.to(self.device)
            else:
                # 假设是 (text, image, target) 或 (input, target)
                if len(batch) == 3:
                    text, image, y = batch
                    text, image, y = text.to(self.device), image.to(self.device), y.to(self.device)
                else:
                    text, y = batch
                    text, y = text.to(self.device), y.to(self.device)
                    image = None
            
            self.optimizer.zero_grad()
            
            # 根据模型类型调用不同方法
            if hasattr(self.model, 'forward_multimodal') and text is not None and image is not None:
                outputs = self.model.forward_multimodal(text, image)
            elif hasattr(self.model, 'forward_text') and text is not None:
                outputs = self.model.forward_text(text)
            elif hasattr(self.model, 'forward_image_only') and image is not None:
                outputs = self.model.forward_image_only(image)
            else:
                # 回退到普通forward
                inputs = text if text is not None else image
                outputs = self.model(inputs, consolidate=consolidate)
            
            loss = loss_fn(outputs['output'], y)
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            if self.scheduler is not None and self.global_step >= self.warmup_steps:
                self.scheduler.step()
            
            total_loss += loss.item() * (y.size(0) if y.dim() == 1 else y.size(0))
            if outputs['output'].dim() > 1 and outputs['output'].size(1) > 1:
                preds = outputs['output'].argmax(dim=-1)
                if y.dim() > 1 and y.size(1) > 1:
                    y = y.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}


# ==================== 数据集工具 ====================

class SimpleDataset(Dataset):
    """简单张量数据集"""
    
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, memory_inputs: Optional[torch.Tensor] = None):
        self.inputs = inputs
        self.targets = targets
        self.memory_inputs = memory_inputs
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = {'input': self.inputs[idx], 'target': self.targets[idx]}
        if self.memory_inputs is not None:
            item['memory_input'] = self.memory_inputs[idx]
        return item


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(
        self,
        texts: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        targets: torch.Tensor = None,
    ):
        self.texts = texts
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {'target': self.targets[idx]}
        if self.texts is not None:
            item['text'] = self.texts[idx]
        if self.images is not None:
            item['image'] = self.images[idx]
        return item


def create_synthetic_dataset(
    num_samples: int = 1000,
    input_dim: int = 512,
    output_dim: int = 10,
    memory_dim: int = 128,
    include_memory: bool = True,
) -> SimpleDataset:
    """创建合成数据集用于测试"""
    inputs = torch.randn(num_samples, input_dim)
    targets = torch.randint(0, output_dim, (num_samples,))
    memory_inputs = torch.randn(num_samples, memory_dim) if include_memory else None
    
    return SimpleDataset(inputs, targets, memory_inputs)


def create_multimodal_dataset(
    num_samples: int = 1000,
    text_dim: int = 512,
    image_size: int = 224,
    output_dim: int = 10,
) -> MultiModalDataset:
    """创建多模态合成数据集"""
    texts = torch.randn(num_samples, text_dim)
    images = torch.randn(num_samples, 3, image_size, image_size)
    targets = torch.randint(0, output_dim, (num_samples,))
    
    return MultiModalDataset(texts, images, targets)