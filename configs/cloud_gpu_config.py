"""
NeuroFlow 1.5B 云GPU训练环境配置
支持 Modal、AWS、阿里云等多种平台

推荐平台:
1. Modal (无服务器GPU) - 最简单，按需付费
2. AWS EC2 (p4d/p5实例) - 高性能，适合大规模训练
3. 阿里云PAI - 国内优化，中文数据友好
"""

import os
import json
from typing import Dict, Optional


# ==================== 硬件需求 ====================

GPU_REQUIREMENTS = {
    "minimum": {
        "gpu": "RTX 3090 / RTX 4090",
        "memory": "24GB",
        "estimated_time": "72小时",
        "cost_estimate": "约 $200-300",
    },
    "recommended": {
        "gpu": "A100 40GB / A100 80GB",
        "memory": "40-80GB",
        "estimated_time": "24-36小时",
        "cost_estimate": "约 $300-500",
    },
    "optimal": {
        "gpu": "H100 80GB",
        "memory": "80GB",
        "estimated_time": "12-18小时",
        "cost_estimate": "约 $400-600",
    },
}


# ==================== Modal 配置 ====================

MODAL_CONFIG = {
    "app_name": "neuroflow-1.5b-training",
    "gpu_type": "A100-40GB",
    "timeout": 86400,  # 24小时
    
    "image": {
        "base": "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        "python": "3.11",
        "packages": [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "datasets>=2.14.0",
            "peft>=0.7.0",
            "trl>=0.7.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.41.0",
            "unsloth",
            "sentencepiece",
            "protobuf",
        ],
    },
    
    "volumes": {
        "data": "/data",
        "checkpoints": "/checkpoints",
        "cache": "/cache",
    },
    
    "secrets": [
        "HF_TOKEN",
        "WANDB_API_KEY",
    ],
}


def get_modal_training_script():
    """生成Modal训练脚本"""
    
    script = '''
import modal

app = modal.App("neuroflow-1.5b-training")

# 定义GPU镜像
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    add_python="3.11"
).pip_install(
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "peft>=0.7.0",
    "trl>=0.7.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.0",
    "unsloth",
    "sentencepiece",
)

# 数据卷
data_volume = modal.Volume.from_name("neuroflow-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("neuroflow-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=86400,  # 24小时
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_neuroflow_1_5b():
    """训练NeuroFlow 1.5B模型"""
    
    import os
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer
    from datasets import load_dataset
    
    # 配置
    config = {
        "base_model": "Qwen/Qwen2-1.5B",
        "lora_r": 64,
        "lora_alpha": 128,
        "max_seq_length": 8192,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
    }
    
    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    
    # 应用LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
    )
    
    # 加载中文数据
    chinese_data = load_dataset("Skywork/Skywork-CN-corpus", split="train")
    
    # 加载代码数据
    code_data = load_dataset("bigcode/the-stack", data_dir="data/python", split="train")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="/checkpoints/neuroflow-1.5b",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=chinese_data,
    )
    
    trainer.train()
    
    # 保存
    model.save_pretrained("/checkpoints/neuroflow-1.5b")
    tokenizer.save_pretrained("/checkpoints/neuroflow-1.5b")
    
    return {"status": "completed", "checkpoints": "/checkpoints/neuroflow-1.5b"}

# 本地入口
@app.local_entrypoint()
def main():
    result = train_neuroflow_1_5b.remote()
    print(f"Training result: {result}")
'''
    
    return script


# ==================== AWS 配置 ====================

AWS_CONFIG = {
    "instance_type": "p4d.24xlarge",  # 8x A100 40GB
    "region": "us-west-2",
    "ami": "ami-0c55b159cbfafe1f0",   # Deep Learning AMI
    
    "setup_commands": [
        "pip install torch transformers datasets peft trl accelerate bitsandbytes unsloth",
        "export HF_TOKEN=YOUR_HF_TOKEN",
        "export WANDB_API_KEY=YOUR_WANDB_KEY",
    ],
    
    "training_command": "python train_1.5b_unsloth.py --epochs 3 --batch_size 16",
    
    "estimated_cost": {
        "p4d.24xlarge": "$32.77/hour",
        "training_time": "24-36 hours",
        "total": "$800-1200",
    },
}


# ==================== 阿里云配置 ====================

ALIYUN_CONFIG = {
    "instance_type": "ecs.gn6v-c8g1.16xlarge",  # V100 32GB
    "region": "cn-hangzhou",
    
    "pai_config": {
        "workspace": "neuroflow-training",
        "resource_type": "GPU",
        "gpu_count": 4,
        "memory": "128GB",
    },
    
    "estimated_cost": {
        "gn6v": "约 ¥20/小时",
        "training_time": "48-72小时",
        "total": "约 ¥1000-1500",
    },
}


# ==================== 本地 Docker 配置 ====================

DOCKER_CONFIG = {
    "image": "neuroflow-training:latest",
    
    "dockerfile": '''
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Python
RUN apt-get update && apt-get install -y python3.11 python3.11-pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 安装依赖
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    transformers>=4.36.0 \
    datasets>=2.14.0 \
    peft>=0.7.0 \
    trl>=0.7.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    unsloth \
    sentencepiece

# 工作目录
WORKDIR /app

# 复制训练脚本
COPY train_1.5b_unsloth.py .
COPY configs/ ./configs/

# 训练命令
CMD ["python", "train_1.5b_unsloth.py", "--epochs", "3", "--batch_size", "16"]
''',
    
    "build_command": "docker build -t neuroflow-training:latest .",
    
    "run_command": "docker run --gpus all -v ./checkpoints:/checkpoints neuroflow-training:latest",
}


def print_hardware_requirements():
    """打印硬件需求"""
    
    print("=" * 70)
    print("NeuroFlow 1.5B Training Hardware Requirements")
    print("=" * 70)
    
    for level, specs in GPU_REQUIREMENTS.items():
        print(f"\n[{level.upper()}]")
        for key, value in specs.items():
            print(f"  {key:15s}: {value}")


def print_platform_configs():
    """打印平台配置"""
    
    print("\n" + "=" * 70)
    print("Cloud Platform Configurations")
    print("=" * 70)
    
    print("\n[Modal - 无服务器GPU]")
    print("  推荐理由: 最简单，无需管理服务器，按秒计费")
    print("  部署命令: modal deploy train_modal.py")
    for key, value in MODAL_CONFIG.items():
        if key not in ["image", "volumes"]:
            print(f"  {key:15s}: {value}")
    
    print("\n[AWS EC2]")
    print("  推荐理由: 高性能，适合大规模分布式训练")
    print("  实例类型:", AWS_CONFIG["instance_type"])
    print("  预估费用:", AWS_CONFIG["estimated_cost"])
    
    print("\n[阿里云 PAI]")
    print("  推荐理由: 国内优化，中文数据友好，速度快")
    print("  实例类型:", ALIYUN_CONFIG["instance_type"])
    print("  预估费用:", ALIYUN_CONFIG["estimated_cost"])


def get_platform_recommendation():
    """获取平台推荐"""
    
    recommendations = {
        "快速开始": {
            "platform": "Modal",
            "reason": "无需配置，一键部署，自动管理GPU",
            "steps": [
                "1. 安装Modal: pip install modal",
                "2. 配置Token: modal token new",
                "3. 部署训练: modal deploy train_modal.py",
            ],
        },
        "大规模训练": {
            "platform": "AWS",
            "reason": "高性能GPU集群，支持分布式训练",
            "steps": [
                "1. 创建EC2实例 (p4d.24xlarge)",
                "2. 安装CUDA和依赖",
                "3. 运行训练脚本",
            ],
        },
        "国内用户": {
            "platform": "阿里云PAI",
            "reason": "国内节点，中文数据加载快",
            "steps": [
                "1. 创建PAI工作空间",
                "2. 配置GPU资源",
                "3. 上传训练脚本运行",
            ],
        },
    }
    
    return recommendations


if __name__ == "__main__":
    print_hardware_requirements()
    print_platform_configs()
    
    print("\n" + "=" * 70)
    print("Platform Recommendations")
    print("=" * 70)
    
    recs = get_platform_recommendation()
    for use_case, rec in recs.items():
        print(f"\n[{use_case}]")
        print(f"  推荐平台: {rec['platform']}")
        print(f"  推荐理由: {rec['reason']}")
        print("  操作步骤:")
        for step in rec["steps"]:
            print(f"    {step}")
    
    # 保存Modal脚本
    modal_script = get_modal_training_script()
    with open("train_modal.py", "w") as f:
        f.write(modal_script)
    print("\n  Modal脚本已保存: train_modal.py")