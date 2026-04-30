
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
