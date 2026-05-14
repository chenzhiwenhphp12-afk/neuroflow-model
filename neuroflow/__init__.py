"""
NeuroFlow — 多模态类脑神经网络框架
====================================
Brain-inspired modular neural network with multimodal capabilities.
Inspired by 2026 neuroscience research.

Author: Chen Zhiwen <chenzhiwenhphp12@gmail.com>
License: MIT
"""

__version__ = "2.1.0"

# ---- C++ Core (preferred - SIMD optimized, 43K params, 0.40ms) ----
try:
    from neuroflow._core import (
        # Tensor
        Tensor,
        TensorOps,
        QuantType,
        MemoryLayout,
        # Layers
        Linear,
        LayerNorm,
        # Brain Networks
        ExecutiveControlNetwork,
        DefaultModeNetwork,
        SalienceNetwork,
        # Memory
        MemoryConsolidationModule,
        LatentKVCache,
        # Model
        NeuroFlowModel,
        NeuroFlowLite,
        ModelConfig,
        ModelOutput,
        ModelStats,
        # Convenience
        create_tensor,
        benchmark,
    )
    _BACKEND = "C++"
    _HAS_CPP = True

except ImportError:
    _BACKEND = "Python"
    _HAS_CPP = False

    # Fallback: pure Python implementations
    from neuroflow.model import NeuroFlowModel as _PyNeuroFlowModel
    from neuroflow.modules import (
        ExecutiveControlNetwork,
        DefaultModeNetwork,
        SalienceNetwork,
        MemoryConsolidationModule,
    )
    from neuroflow.model_lite import NeuroFlowModelLite

    # Alias Python model to match C++ API
    NeuroFlowModel = _PyNeuroFlowModel
    NeuroFlowLite = NeuroFlowModelLite

# ---- Python Modules (optional — requires torch) ----
try:
    from neuroflow.deepseek_optimizations import (
        LatentKVCompression,
        SparseMoE,
        RotaryPositionalEmbedding,
        QuantizedLinear,
        quantize_model,
        EfficientMemoryModule,
        OptimizedECN,
    )
    _HAS_DEEPSEEK = True
except ImportError:
    _HAS_DEEPSEEK = False
    LatentKVCompression = None
    SparseMoE = None
    RotaryPositionalEmbedding = None
    QuantizedLinear = None
    quantize_model = None
    EfficientMemoryModule = None
    OptimizedECN = None

try:
    from neuroflow.deepseek_v4_optimizations import (
        EngramMemory,
        EngramConfig,
        MuonOptimizer,
        FlashAttentionV4,
        RotaryPositionalEmbeddingV4,
        MultiHyperConnection,
        IterativeSelfCorrection,
        EAGLESpeculativeDecoding,
        NeuroFlowV4,
    )
    _HAS_DEEPSEEK_V4 = True
except ImportError:
    _HAS_DEEPSEEK_V4 = False
    EngramMemory = None
    EngramConfig = None
    MuonOptimizer = None
    FlashAttentionV4 = None
    RotaryPositionalEmbeddingV4 = None
    MultiHyperConnection = None
    IterativeSelfCorrection = None
    EAGLESpeculativeDecoding = None
    NeuroFlowV4 = None


def get_backend():
    """Return the active backend: 'C++' or 'Python'."""
    return _BACKEND


from neuroflow.decoder import TextDecoder, TokenSampler

__all__ = [
    # Version & backend
    "__version__",
    "get_backend",
    # C++ Core / Python Fallback
    "NeuroFlowModel",
    "NeuroFlowLite",
    "ExecutiveControlNetwork",
    "DefaultModeNetwork",
    "SalienceNetwork",
    "MemoryConsolidationModule",
    # DeepSeek V3 Optimizations
    "LatentKVCompression",
    "SparseMoE",
    "RotaryPositionalEmbedding",
    "QuantizedLinear",
    "quantize_model",
    "EfficientMemoryModule",
    "OptimizedECN",
    # DeepSeek V4 Optimizations
    "EngramMemory",
    "EngramConfig",
    "MuonOptimizer",
    "FlashAttentionV4",
    "RotaryPositionalEmbeddingV4",
    "MultiHyperConnection",
    "IterativeSelfCorrection",
    "EAGLESpeculativeDecoding",
    "NeuroFlowV4",
]
