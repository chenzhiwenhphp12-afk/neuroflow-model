"""
NeuroFlow V2 - 低算力优化模块

基于 DeepSeek V3 技术实现:
1. MLA (Multi-head Latent Attention) - 低秩压缩记忆
2. Sparse Expert Routing - 稀疏专家路由
3. KV-Cache Memory - 增量记忆缓存
4. Dynamic Quantization - 动态量化支持

核心目标: 低算力运行 + 长记忆 + 快速响应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List


class LatentCompressedAttention(nn.Module):
    """
    MLA (Multi-head Latent Attention) 低秩压缩实现
    
    DeepSeek 核心创新: KV 压缩到潜在空间，推理时只需缓存少量向量
    内存占用: O(d_kv) 而非 O(n_heads * d_head)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,  # 低秩压缩维度 (DeepSeek: 512 for large model)
        num_heads: int = 4,
        q_head_dim: int = 32,
        kv_head_dim: int = 32,
        rope_dim: int = 16,  # RoPE 位置编码维度
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_head_dim = q_head_dim
        self.kv_head_dim = kv_head_dim
        self.rope_dim = rope_dim
        self.latent_dim = latent_dim
        
        # Q 投影 (不压缩)
        self.q_proj = nn.Linear(input_dim, num_heads * q_head_dim, bias=False)
        
        # KV 压缩到潜在空间 (MLA 核心)
        self.kv_compress = nn.Linear(input_dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, num_heads * kv_head_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, num_heads * kv_head_dim, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(num_heads * kv_head_dim, input_dim, bias=False)
        
        # RoPE (旋转位置编码)
        self.rope_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / rope_dim)),
            requires_grad=False
        )
        
        # KV Cache (长记忆核心)
        self.kv_cache: Optional[torch.Tensor] = None  # (max_seq, latent_dim)
        self.cache_len = 0
        
    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """旋转位置编码 (仅应用于 rope_dim 部分)"""
        # x: (batch, seq, dim)
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]
        
        # 旋转
        freqs = positions.float() * self.rope_freq  # (seq, rope_dim/2)
        cos = freqs.cos().unsqueeze(0)  # (1, seq, rope_dim/2)
        sin = freqs.sin().unsqueeze(0)
        
        x1, x2 = x_rope.chunk(2, dim=-1)
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return torch.cat([rotated, x_pass], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        max_cache_len: int = 512,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        MLA 前向传播
        
        Args:
            x: (batch, seq_len, input_dim)
            use_cache: 是否启用 KV Cache (长记忆)
            max_cache_len: 最大缓存长度
        
        Returns:
            output: (batch, seq_len, input_dim)
            cache_info: 缓存状态信息
        """
        batch, seq_len, _ = x.shape
        
        # Q 投影
        q = self.q_proj(x)  # (batch, seq, num_heads * q_head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim)
        
        # KV 压缩 (MLA 核心 - 低秩)
        kv_latent = self.kv_compress(x)  # (batch, seq, latent_dim) <- 只缓存这个!
        
        # 从潜在空间解压 K, V
        k = self.k_up(kv_latent)  # (batch, seq, num_heads * kv_head_dim)
        v = self.v_up(kv_latent)
        k = k.view(batch, seq_len, self.num_heads, self.kv_head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.kv_head_dim)
        
        # 应用 RoPE 到 Q, K 的部分维度
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # Q 应用 RoPE (仅 rope_dim 部分)
        q_rope = q[..., :self.rope_dim]
        q_rope = self.apply_rope(q_rope.unsqueeze(0), positions)
        q = torch.cat([q_rope.squeeze(0), q[..., self.rope_dim:]], dim=-1)
        
        # KV Cache 管理 (长记忆)
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = torch.zeros(max_cache_len, self.latent_dim, device=x.device)
            
            # 增量更新缓存 (只存 latent，不存完整 KV!)
            new_len = min(self.cache_len + seq_len, max_cache_len)
            if self.cache_len < max_cache_len:
                self.kv_cache[self.cache_len:new_len] = kv_latent[0, :new_len - self.cache_len]
                self.cache_len = new_len
            
            # 从缓存重建历史 KV
            cached_latent = self.kv_cache[:self.cache_len]  # (cache_len, latent_dim)
            k_cached = self.k_up(cached_latent).view(self.cache_len, self.num_heads, self.kv_head_dim)
            v_cached = self.v_up(cached_latent).view(self.cache_len, self.num_heads, self.kv_head_dim)
            
            # 合并当前和历史
            k_full = torch.cat([k_cached.unsqueeze(0), k], dim=1)  # (1, cache+seq, heads, dim)
            v_full = torch.cat([v_cached.unsqueeze(0), v], dim=1)
        else:
            k_full = k
            v_full = v
        
        # Attention 计算
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
        
        # Flash Attention 风格 (scaled dot-product)
        scale = 1.0 / math.sqrt(self.q_head_dim)
        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output = torch.matmul(attn_weights, v_full)  # (batch, heads, seq, kv_dim)
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out_proj(output)
        
        return output, {
            "cache_len": self.cache_len if use_cache else 0,
            "latent_cached": self.latent_dim * self.cache_len if use_cache else 0,  # 实际缓存大小
            "kv_full_size": self.num_heads * self.kv_head_dim * self.cache_len if use_cache else 0,  # 不压缩的大小
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.kv_cache = None
        self.cache_len = 0


class SparseExpertRouter(nn.Module):
    """
    稀疏专家路由 (DeepSeek MoE 风格)
    
    核心思想: 大模型中只激活少量专家，降低推理计算量
    DeepSeek V3: 671B 参数，每个 token 只激活 37B (6/64 专家)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_dim: int = 128,
        num_activated: int = 2,  # 每个 token 激活的专家数
        shared_experts: int = 1,  # 共享专家 (始终激活)
        score_func: str = "sigmoid",  # DeepSeek 用 sigmoid 避免 softmax 的负载不平衡
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.shared_experts = shared_experts
        self.expert_dim = expert_dim
        
        # 路由门控
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.score_func = score_func
        
        # 专家网络 (轻量 MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, input_dim),
            )
            for _ in range(num_experts)
        ])
        
        # 共享专家 (始终激活，提供基础能力)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, expert_dim * shared_experts),
            nn.GELU(),
            nn.Linear(expert_dim * shared_experts, input_dim),
        )
        
        # 路由缩放因子 (DeepSeek 风格)
        self.route_scale = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        稀疏专家路由
        
        Args:
            x: (batch, seq, input_dim) 或 (batch, input_dim)
        
        Returns:
            output: 专家混合输出
            routing_info: 路由统计信息
        """
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)
        batch, seq_len, dim = x.shape
        
        # 计算路由分数
        gate_scores = self.gate(x)  # (batch, seq, num_experts)
        
        if self.score_func == "sigmoid":
            # Sigmoid 路由 (DeepSeek 风格 - 更稳定的负载均衡)
            scores = torch.sigmoid(gate_scores) * self.route_scale
        else:
            scores = F.softmax(gate_scores, dim=-1)
        
        # Top-K 选择
        topk_scores, topk_indices = scores.topk(self.num_activated, dim=-1)
        
        # 归一化 top-k 分数
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        # 专家计算 (稀疏激活)
        expert_outputs = torch.zeros_like(x)
        activated_counts = torch.zeros(self.num_experts, device=x.device)
        
        # 并行专家计算 (优化: 使用 einsum 或 scatter)
        for i in range(self.num_activated):
            expert_idx = topk_indices[..., i]  # (batch, seq)
            weight = topk_weights[..., i].unsqueeze(-1)  # (batch, seq, 1)
            
            # 获取专家输出
            # 优化: 避免循环，使用批量索引
            flat_idx = expert_idx.view(-1)
            flat_x = x.view(-1, dim)
            
            # 每个专家处理分配给它的 token
            for e_idx in range(self.num_experts):
                mask = (flat_idx == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](flat_x[mask])
                    # scatter 回原位置
                    out_idx = mask.nonzero().squeeze(-1)
                    expert_outputs.view(-1, dim)[out_idx] += (
                        weight.view(-1, 1)[out_idx] * expert_out
                    )
                    activated_counts[e_idx] += 1
        
        # 共享专家 (始终激活)
        shared_out = self.shared(x)
        output = expert_outputs + shared_out
        
        # 计算效率统计
        total_params = sum(
            sum(p.numel() for p in e.parameters()) 
            for e in self.experts
        )
        activated_params = (
            self.num_activated * total_params // self.num_experts 
            + sum(p.numel() for p in self.shared.parameters())
        )
        
        return output.reshape(original_shape), {
            "activated_ratio": activated_params / total_params,
            "expert_usage": activated_counts / activated_counts.sum(),
            "topk_indices": topk_indices,
        }


class HierarchicalMemoryBank(nn.Module):
    """
    层级化记忆库 (长记忆核心)
    
    设计:
    - 短期记忆: 快速访问，少量 slot
    - 长期记忆: 大容量，低频访问
    - 压缩存储: 类似 MLA 的潜在压缩
    """
    
    def __init__(
        self,
        input_dim: int,
        short_term_slots: int = 32,  # 短期记忆槽
        long_term_slots: int = 256,  # 长期记忆槽
        latent_dim: int = 64,  # 压缩维度
        compression_ratio: float = 0.5,  # 长期记忆压缩比
    ):
        super().__init__()
        self.input_dim = input_dim
        self.short_term_slots = short_term_slots
        self.long_term_slots = long_term_slots
        
        # 短期记忆 (不压缩，快速访问)
        self.short_term_bank = nn.Parameter(
            torch.randn(short_term_slots, input_dim) * 0.02
        )
        self.short_term_usage = torch.zeros(short_term_slots)
        
        # 长期记忆 (压缩存储)
        self.long_term_bank = nn.Parameter(
            torch.randn(long_term_slots, latent_dim) * 0.02
        )
        self.long_term_decoder = nn.Linear(latent_dim, input_dim)
        
        # 记忆重要性追踪 (用于遗忘机制)
        self.importance_scores = nn.Parameter(
            torch.zeros(long_term_slots)
        )
        
        # 编码/检索
        self.query_proj = nn.Linear(input_dim, latent_dim)
        self.short_query_proj = nn.Linear(input_dim, input_dim)
        
        # 记忆衰减率 (模拟遗忘曲线)
        self.decay_rate = 0.01
        
    def encode_to_long_term(self, x: torch.Tensor) -> torch.Tensor:
        """编码到长期记忆的潜在空间"""
        return self.query_proj(x)
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k_short: int = 4,
        top_k_long: int = 8,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        层级化检索
        
        Args:
            query: (batch, input_dim)
            top_k_short: 短期记忆检索数
            top_k_long: 长期记忆检索数
        
        Returns:
            retrieved: 检索结果
            retrieval_info: 检索统计
        """
        batch = query.size(0)
        
        # 短期记忆检索 (快速，不压缩)
        short_query = self.short_query_proj(query)
        short_scores = F.softmax(
            torch.matmul(short_query, self.short_term_bank.T) / 
            math.sqrt(self.input_dim), dim=-1
        )
        short_topk_scores, short_topk_idx = short_scores.topk(top_k_short, dim=-1)
        short_retrieved = torch.matmul(short_topk_scores, 
            self.short_term_bank[short_topk_idx])
        
        # 长期记忆检索 (压缩，大容量)
        long_query = self.query_proj(query)  # (batch, latent_dim)
        long_scores = F.softmax(
            torch.matmul(long_query, self.long_term_bank.T) /
            math.sqrt(self.long_term_bank.size(-1)), dim=-1
        )
        long_topk_scores, long_topk_idx = long_scores.topk(top_k_long, dim=-1)
        
        # 解压长期记忆
        long_latent = self.long_term_bank[long_topk_idx]  # (batch, top_k, latent)
        long_retrieved = self.long_term_decoder(long_latent)  # (batch, top_k, input)
        long_retrieved = torch.matmul(long_topk_scores.unsqueeze(-1), long_retrieved)
        
        # 合并短期和长期
        retrieved = short_retrieved + long_retrieved
        
        # 更新重要性分数
        with torch.no_grad():
            used_indices = long_topk_idx.unique()
            self.importance_scores[used_indices] += 0.1
        
        return retrieved, {
            "short_topk": short_topk_idx,
            "long_topk": long_topk_idx,
            "short_scores": short_topk_scores,
            "long_scores": long_topk_scores,
        }
    
    def consolidate(
        self,
        x: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ):
        """
        记忆巩固 (短期 → 长期迁移)
        
        Args:
            x: 要巩固的记忆
            importance: 记忆重要性权重
        """
        with torch.no_grad():
            # 编码到长期记忆
            encoded = self.encode_to_long_term(x)
            
            # 找到最不重要的槽位进行替换
            if importance is not None:
                replace_idx = self.importance_scores.argmin()
            else:
                replace_idx = torch.randint(0, self.long_term_slots, (1,)).item()
            
            # 更新长期记忆
            self.long_term_bank[replace_idx] = encoded.mean(0)
            
            # 应用遗忘衰减
            self.importance_scores *= (1 - self.decay_rate)


class QuantizedLinear(nn.Module):
    """
    动态量化线性层 (DeepSeek FP8 风格)
    
    简化实现: 使用 PyTorch 内置量化，不依赖 Triton
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quantize: bool = True,
        block_size: int = 128,
    ):
        super().__init__()
        self.quantize = quantize
        self.block_size = block_size
        
        # 权重存储为 fp16，推理时量化
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化 scale (动态计算)
        self.scale = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize:
            return F.linear(x, self.weight, self.bias)
        
        # 动态量化 (推理时)
        with torch.no_grad():
            # Block-wise quantization
            weight_flat = self.weight.data
            scale = weight_flat.abs().max() / 127.0
            self.scale = scale
        
        # 量化计算 (使用 int8 模拟)
        # 实际实现会用 FP8，这里简化为模拟
        weight_q = (self.weight / self.scale).round() * self.scale
        
        return F.linear(x, weight_q, self.bias)


class OptimizedECN(nn.Module):
    """
    优化版执行控制网络
    
    整合:
    - Sparse Expert Routing (稀疏专家)
    - Latent Compressed Attention (低秩压缩)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        activated_experts: int = 1,
    ):
        super().__init__()
        
        # Sparse Expert MLP
        self.expert_mlp = SparseExpertRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            expert_dim=hidden_dim,
            num_activated=activated_experts,
        )
        
        # MLA Attention
        self.attention = LatentCompressedAttention(
            input_dim=hidden_dim,
            latent_dim=hidden_dim // 4,
            num_heads=4,
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.value_proj = nn.Linear(hidden_dim, 1)
        
        self.hidden_dim = hidden_dim
        
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: (batch, input_dim) 或 (batch, seq, input_dim)
        
        Returns:
            decision: (batch, output_dim)
            value: (batch, 1)
            routing_info: 专家路由信息
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)
        
        # Sparse Expert Routing
        expert_out, routing_info = self.expert_mlp(x)
        
        # MLA Attention (可选缓存)
        attn_out, cache_info = self.attention(expert_out, use_cache=use_cache)
        
        # 合并
        combined = expert_out + attn_out
        
        # 输出
        decision = self.output_proj(combined.squeeze(1))
        value = self.value_proj(combined.squeeze(1))
        
        return decision, value, {
            "routing": routing_info,
            "cache": cache_info,
        }


class OptimizedDMN(nn.Module):
    """
    优化版默认模式网络
    
    整合:
    - Hierarchical Memory (层级化记忆)
    - Latent Compression
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        memory_slots: int = 256,
    ):
        super().__init__()
        
        # 层级化记忆库
        self.memory = HierarchicalMemoryBank(
            input_dim=input_dim,
            short_term_slots=32,
            long_term_slots=memory_slots,
            latent_dim=latent_dim,
        )
        
        # 创造性生成 (轻量 MLP)
        self.generator = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # 输出
        self.vision_proj = nn.Linear(latent_dim, input_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        retrieve_top_k: int = 8,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch, input_dim)
        
        Returns:
            vision: (batch, input_dim) - 创造性联想
            memory_info: 记忆检索信息
        """
        # 层级化记忆检索
        retrieved, retrieval_info = self.memory.retrieve(
            x, top_k_short=4, top_k_long=retrieve_top_k
        )
        
        # 生成创造性联想
        vision = self.generator(retrieved)
        vision = self.vision_proj(vision)
        
        return vision, {
            "retrieval": retrieval_info,
        }


class FastInferenceCache:
    """
    快速推理缓存
    
    目标: 最小化推理延迟
    - 预计算常见模式
    - 懒加载权重
    - 增量更新
    """
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self.pattern_cache: Dict[str, torch.Tensor] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取缓存"""
        if key in self.pattern_cache:
            self.hit_count += 1
            return self.pattern_cache[key]
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: torch.Tensor):
        """存储缓存"""
        if len(self.pattern_cache) >= self.cache_size:
            # LRU 风格: 随机删除
            oldest = list(self.pattern_cache.keys())[0]
            del self.pattern_cache[oldest]
        self.pattern_cache[key] = value.detach()
        
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# 导出优化模块
__all__ = [
    'LatentCompressedAttention',
    'SparseExpertRouter',
    'HierarchicalMemoryBank',
    'QuantizedLinear',
    'OptimizedECN',
    'OptimizedDMN',
    'FastInferenceCache',
]