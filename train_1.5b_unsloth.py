"""
NeuroFlow 1.5B 训练脚本
使用 Unsloth + LoRA 高效微调
中文理解 + 编程能力 专精训练

运行方式:
    python train_1.5b_unsloth.py --config configs/train_1.5b.yaml

硬件需求:
    - 最小: RTX 3090 (24GB)
    - 推荐: A100 (40GB) / RTX 4090
"""

import os
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth未安装，请运行: pip install unsloth")

try:
    from transformers import TrainingArguments, Trainer
    from datasets import load_dataset, Dataset
    from peft import LoraConfig, get_peft_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers未安装，请运行: pip install transformers datasets peft")


@dataclass
class NeuroFlowTrainingConfig:
    """NeuroFlow 1.5B 训练配置"""
    
    # ==================== 基础模型 ====================
    base_model: str = "Qwen/Qwen2-1.5B"  # 基础中文模型
    
    # ==================== LoRA配置 ====================
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # ==================== 序列长度 ====================
    max_seq_length: int = 8192
    
    # ==================== 训练参数 ====================
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # ==================== 优化器 ====================
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"
    
    # ==================== 精度 ====================
    use_bfloat16: bool = True
    use_gradient_checkpointing: bool = True
    
    # ==================== 数据 ====================
    chinese_data_ratio: float = 0.6
    code_data_ratio: float = 0.4
    data_dir: str = "./data"
    
    # ==================== 输出 ====================
    output_dir: str = "./checkpoints/neuroflow-1.5b-chinese-code"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10
    
    # ==================== 评估 ====================
    eval_strategy: str = "epoch"
    eval_steps: int = 500


def load_model_with_lora(config: NeuroFlowTrainingConfig):
    """加载模型并应用LoRA"""
    
    if UNSLOTH_AVAILABLE:
        # 使用Unsloth加载（更快、更省内存）
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=None,  # 自动检测
            load_in_4bit=True,  # 4bit量化加载
        )
        
        # 应用LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        return model, tokenizer
    
    else:
        # 使用标准transformers加载
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16 if config.use_bfloat16 else torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # 应用PEFT LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer


def prepare_training_arguments(config: NeuroFlowTrainingConfig):
    """准备训练参数"""
    
    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        bf16=config.use_bfloat16 and is_bfloat16_supported(),
        fp16=not config.use_bfloat16,
        gradient_checkpointing=config.use_gradient_checkpointing,
        optim=config.optimizer,
        lr_scheduler_type=config.lr_scheduler,
        report_to="none",  # 可改为 "wandb"
    )
    
    return args


def load_training_data(config: NeuroFlowTrainingConfig, tokenizer):
    """加载训练数据"""
    
    # 中文数据
    chinese_datasets = [
        "Skywork/Skywork-CN-corpus",
        "shibing624/multi-cn-nli",
    ]
    
    # 编程数据
    code_datasets = [
        "bigcode/the-stack",
        "codeparrot/codeparrot-clean",
    ]
    
    # 中文编程混合
    chinese_code_datasets = [
        "leetcode-cn-solutions",
    ]
    
    all_data = []
    
    # 加载中文数据
    for ds_name in chinese_datasets:
        try:
            ds = load_dataset(ds_name, split="train")
            # 按比例采样
            sample_size = int(len(ds) * config.chinese_data_ratio / len(chinese_datasets))
            ds = ds.select(range(min(sample_size, len(ds))))
            all_data.append(ds)
        except Exception as e:
            print(f"加载 {ds_name} 失败: {e}")
    
    # 加载代码数据
    for ds_name in code_datasets:
        try:
            ds = load_dataset(ds_name, split="train", data_dir="data/python")
            sample_size = int(len(ds) * config.code_data_ratio / len(code_datasets))
            ds = ds.select(range(min(sample_size, len(ds))))
            all_data.append(ds)
        except Exception as e:
            print(f"加载 {ds_name} 失败: {e}")
    
    # 合并数据集
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(all_data)
    
    # 预处理
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
    
    tokenized_dataset = combined_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=combined_dataset.column_names,
    )
    
    return tokenized_dataset


def train_model(config: NeuroFlowTrainingConfig):
    """训练模型"""
    
    print("=" * 70)
    print("NeuroFlow 1.5B Training - Chinese + Code")
    print("=" * 70)
    
    print("\n[配置信息]")
    print(f"  基础模型:       {config.base_model}")
    print(f"  LoRA r:         {config.lora_r}")
    print(f"  Max Seq Len:    {config.max_seq_length}")
    print(f"  Epochs:         {config.num_epochs}")
    print(f"  Batch Size:     {config.batch_size}")
    print(f"  Learning Rate:  {config.learning_rate}")
    print(f"  Output Dir:     {config.output_dir}")
    
    # 加载模型
    print("\n[加载模型]")
    model, tokenizer = load_model_with_lora(config)
    
    # 查看可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = 100 * trainable_params / all_params
    
    print(f"  可训练参数:     {trainable_params:,} ({trainable_ratio:.2f}%)")
    print(f"  总参数:         {all_params:,}")
    
    # 加载数据
    print("\n[加载数据]")
    train_dataset = load_training_data(config, tokenizer)
    print(f"  训练样本:       {len(train_dataset):,}")
    
    # 训练参数
    training_args = prepare_training_arguments(config)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n[开始训练]")
    trainer.train()
    
    # 保存模型
    print("\n[保存模型]")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 导出GGUF格式（可选）
    if UNSLOTH_AVAILABLE:
        gguf_path = os.path.join(config.output_dir, "neuroflow-1.5b.gguf")
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"  GGUF模型:       {gguf_path}")
    
    print("\n[训练完成]")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="NeuroFlow 1.5B Training")
    parser.add_argument("--config", type=str, default="configs/train_1.5b.yaml")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/neuroflow-1.5b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    
    args = parser.parse_args()
    
    config = NeuroFlowTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
    )
    
    train_model(config)


if __name__ == "__main__":
    main()