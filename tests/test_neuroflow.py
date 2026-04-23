"""NeuroFlow 基础测试"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflow.model import NeuroFlowModel
from neuroflow.modules import (
    ExecutiveControlNetwork,
    DefaultModeNetwork,
    SalienceNetwork,
    MemoryConsolidationModule,
)


def test_model_forward():
    """测试模型前向传播"""
    model = NeuroFlowModel(input_dim=64, hidden_dim=32, output_dim=5)
    x = torch.randn(4, 64)
    result = model(x)

    assert "output" in result
    assert result["output"].shape == (4, 5)
    assert "saliency" in result
    assert "decision" in result
    assert "value" in result
    print("✓ test_model_forward passed")


def test_model_manifold():
    """测试流形轨迹计算"""
    model = NeuroFlowModel(input_dim=64, hidden_dim=32, output_dim=5)
    x = torch.randn(2, 64)
    trajectory = model.get_manifold_trajectory(x, steps=5)

    assert trajectory.shape == (2, 5, 32)
    print("✓ test_model_manifold passed")


def test_modules():
    """测试各核心模块"""
    # ECN
    ecn = ExecutiveControlNetwork(input_dim=32, hidden_dim=16, output_dim=4)
    x = torch.randn(2, 32)
    decision, value, hidden = ecn(x)
    assert decision.shape == (2, 4)
    assert value.shape == (2, 1)

    # SN
    sn = SalienceNetwork(input_dim=32, hidden_dim=16)
    saliency, gates, anomaly = sn(x)
    assert saliency.shape == (2, 1)
    assert gates.shape == (2, 2)

    # Memory
    mem = MemoryConsolidationModule(input_dim=32, memory_slots=8, memory_dim=16)
    retrieved, attention = mem(x)
    assert retrieved.shape == (2, 32)
    assert attention.shape == (2, 8)

    # DMN
    dmn = DefaultModeNetwork(memory_dim=16, latent_dim=8)
    mem_input = torch.randn(2, 16)
    vision, associations, latent = dmn(mem_input)
    assert latent.shape == (2, 8)

    print("✓ test_modules passed")


def test_memory_consolidation():
    """测试记忆巩固"""
    mem = MemoryConsolidationModule(input_dim=32, memory_slots=8, memory_dim=16)
    initial_bank = mem.memory_bank.clone()

    x = torch.randn(4, 32)
    mem.consolidate(x)

    # 记忆槽应该已更新
    assert not torch.allclose(initial_bank, mem.memory_bank)
    print("✓ test_memory_consolidation passed")


def test_consistency():
    """测试模型一致性（相同输入应产生相同输出）"""
    model = NeuroFlowModel(input_dim=64, hidden_dim=32, output_dim=5)
    model.eval()

    x = torch.randn(2, 64)
    with torch.no_grad():
        r1 = model(x)
        r2 = model(x)

    assert torch.allclose(r1["output"], r2["output"])
    print("✓ test_consistency passed")


if __name__ == "__main__":
    test_model_forward()
    test_model_manifold()
    test_modules()
    test_memory_consolidation()
    test_consistency()
    print("\n✅ All tests passed!")
