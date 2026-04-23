"""
NeuroFlow 蒸馏工具

包含:
1. MLPTeacher: 一个简单的 MLP 作为教师模型 (用于演示蒸馏机制)
2. LLMTeacher: 接入 LLM API (如 Qwen) 获取软标签的类
3. PrecomputedTeacher: 从文件加载预计算的软标签
"""

import torch
import torch.nn as nn
import json
import os
import requests

# -----------------------------------------------------------
# 本地教师模型
# -----------------------------------------------------------

class MLPTeacher(nn.Module):
    """
    一个简单的多层感知机作为教师模型。
    通常教师模型比学生模型更宽、更深，拥有更强的拟合能力。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_teacher(model, X, y, epochs=50, lr=0.001):
    """快速训练教师模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  [Teacher] Epoch {epoch}, Loss: {total_loss:.4f}")


class TeacherWrapper:
    """封装本地模型，适配蒸馏循环"""
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))


# -----------------------------------------------------------
# 真实大模型教师 (Qwen / DashScope)
# -----------------------------------------------------------

class LLMTeacher:
    """
    使用 DashScope (Qwen) API 作为教师。
    适用于文本或图像分类任务。
    注意：API 调用较慢，建议配合 PrecomputedTeacher 使用。
    """
    def __init__(self, api_key=None, model="qwen-turbo", num_classes=10):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY. Set it in .env or pass api_key.")
        
        self.model = model
        self.num_classes = num_classes
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def get_logits(self, inputs):
        """
        inputs: 字符串描述 (例如: "Digits image showing number 3...")
        返回: tensor of shape (1, num_classes)
        """
        # 构造 Prompt，要求输出概率分布
        prompt = (
            f"Please classify the following input into one of {self.num_classes} classes (0 to {self.num_classes-1}). "
            f"Return ONLY a JSON list of probabilities summing to 1.0. Example: [0.1, 0.8, ...]\n\n"
            f"Input: {inputs}"
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful classifier assistant that outputs JSON probability lists."},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        
        try:
            resp = requests.post(self.base_url, headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            text = result["output"]["text"]
            
            # 解析 JSON
            import re
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                probs = json.loads(json_match.group())
                logits = torch.tensor(probs).unsqueeze(0) # (1, C)
                return logits
            else:
                print(f"[LLMTeacher] Failed to parse JSON from: {text}")
                return torch.zeros(1, self.num_classes)
        except Exception as e:
            print(f"[LLMTeacher] API Error: {e}")
            return torch.zeros(1, self.num_classes)


# -----------------------------------------------------------
# 预计算软标签 (用于高速蒸馏)
# -----------------------------------------------------------

class PrecomputedTeacher:
    """
    加载预计算好的 Soft Labels 文件 (.pt)
    文件应包含一个 tensor: (N, num_classes)
    """
    def __init__(self, logits_path, indices_map=None):
        """
        logits_path: 指向 .pt 文件的路径
        indices_map: 数据集索引到 logits 行索引的映射 (如果顺序一致可省略)
        """
        self.logits = torch.load(logits_path, map_location="cpu")
        self.indices_map = indices_map

    def __call__(self, x, indices=None):
        """
        x: 当前 batch 的输入 (忽略，仅做兼容)
        indices: 当前 batch 在原始数据集中的索引列表
        """
        if indices is None:
            # 如果没有提供索引，假设是顺序取出的前 N 个
            batch_size = x.shape[0]
            indices = list(range(batch_size)) # 这通常需要外层循环配合
            
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        batch_logits = self.logits[indices]
        return batch_logits
