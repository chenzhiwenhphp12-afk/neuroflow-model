"""
NeuroFlow 蒸馏工具

包含:
1. MLPTeacher: 一个简单的 MLP 作为教师模型 (用于演示蒸馏机制)
2. LLMTeacher: 预留接口，用于接入 LLM API (如 Qwen, Llama)
"""

import torch
import torch.nn as nn


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
    """
    封装教师模型，适配蒸馏循环的调用接口。
    支持直接传入模型或函数。
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))
