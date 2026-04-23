"""
预计算 LLM 教师软标签 (Soft Labels)

使用 DashScope (Qwen) API 为数据集生成概率分布。
结果保存为 .pt 文件，供 train.py 的 --teacher-logits-file 使用。

用法:
    export DASHSCOPE_API_KEY="sk-..."
    python scripts/precompute_logits.py --dataset digits --model qwen-turbo
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.teacher import LLMTeacher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--output", default="teacher_logits.pt")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY not set. Please export it.")
        return

    # 1. 加载数据集
    if args.dataset == "digits":
        from scripts.train import load_digits_dataset
        (X_train, y_train), _ = load_digits_dataset()
    elif args.dataset == "synthetic":
        from scripts.train import load_synthetic_dataset
        X_train, y_train = load_synthetic_dataset()
    else:
        raise ValueError("Unsupported dataset")

    num_classes = 10
    print(f"[Precompute] Loaded {len(X_train)} samples for {args.dataset}")

    # 2. 初始化 LLM 教师
    teacher = LLMTeacher(model=args.model, num_classes=num_classes)

    # 3. 预计算
    all_logits = torch.zeros(len(X_train), num_classes)

    # 为了演示，我们只取前 50 个样本进行预计算
    # 在实际使用中，您可以调整 sample_limit 或移除它以处理全部数据
    sample_limit = 50 
    print(f"[Precompute] Generating logits for first {sample_limit} samples...")

    for i in tqdm(range(sample_limit)):
        x = X_train[i]
        
        # 将输入转换为文本描述
        if args.dataset == "digits":
            # 8x8 图像转为简单的 ASCII 描述
            img = x.view(8, 8).numpy()
            desc = f"8x8 grid of pixel intensities:\n{np.array2string(img, precision=1, separator=', ')}"
        else:
            desc = str(x.numpy())
            
        logits = teacher.get_logits(desc)
        all_logits[i] = logits.squeeze(0)

    # 4. 保存
    torch.save(all_logits, args.output)
    print(f"[Precompute] Saved logits to {args.output} (Shape: {all_logits.shape})")
    print(f"[Precompute] Sample distribution for item 0: {all_logits[0]}")

if __name__ == "__main__":
    main()
