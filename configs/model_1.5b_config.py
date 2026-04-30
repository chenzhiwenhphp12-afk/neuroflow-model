"""
NeuroFlow 1.5B 参数模型配置
专精: 中文理解 + 编程能力

基于类脑架构 + Transformer + MLA压缩
"""

import math

class NeuroFlow1_5BConfig:
    """1.5B参数模型配置 - 中文+编程专精"""
    
    # ==================== 基础模型 ====================
    model_type = "neuroflow-1.5b-chinese-code"
    vocab_size = 60000          # 中文词汇 + 代码词汇
    hidden_dim = 2048           # 隐藏维度
    intermediate_dim = 8192     # FFN中间层 (4x hidden)
    num_attention_heads = 32    # 注意力头
    head_dim = 64               # 每头维度
    
    # ==================== 层数配置 ====================
    num_layers = 24             # Transformer层数
    
    # ==================== 类脑网络 ====================
    ecn_layers = 10            # 执行控制网络层数
    dmn_associations = 16      # 默认模式网络联想头
    sn_hidden_dim = 1024       # 显著性网络隐藏层
    
    # ==================== 记忆系统 ====================
    memory_slots = 512          # 记忆槽 (扩展)
    memory_dim = 768            # 记忆维度
    ltp_rate = 0.01             # LTP学习率
    
    # ==================== MLA压缩 ====================
    use_mla = True
    mla_latent_dim = 192        # MLA压缩维度
    mla_num_heads = 40
    max_cache_len = 8192        # 最大缓存长度
    
    # ==================== 中文理解专精 ====================
    chinese_vocab_size = 50000  # 中文词汇
    code_vocab_size = 15000     # 代码词汇
    max_seq_len = 8192          # 最大序列长度
    chinese_tokenizer = "bpe"   # BPE分词
    position_encoding = "rope"  # 旋转位置编码
    
    # 中文任务专精
    chinese_tasks = [
        "语义理解",
        "情感分析",
        "文本生成",
        "问答系统",
        "摘要生成",
        "机器翻译",
        "阅读理解",
    ]
    
    # ==================== 编程能力专精 ====================
    code_languages = [
        "Python", "JavaScript", "Java", "C++", "Go",
        "Rust", "TypeScript", "SQL", "Shell", "HTML/CSS",
    ]
    
    code_tasks = [
        "代码生成",
        "代码补全",
        "代码解释",
        "错误修复",
        "代码优化",
        "API设计",
        "算法实现",
        "单元测试",
    ]
    
    # ==================== 多模态 ====================
    multimodal = True
    text_dim = 2048
    image_size = 224
    vision_hidden_dim = 1024    # vision维度
    vision_num_layers = 10     # vision层数
    fusion_dim = 2048
    
    # ==================== 量化 ====================
    quantization = True
    quant_type = "int8"
    
    # ==================== 训练参数 ====================
    dropout = 0.1
    layer_norm_eps = 1e-5
    gradient_checkpointing = True
    
    def calculate_params(self):
        """计算总参数量"""
        params = {}
        
        # Embedding
        params['embedding'] = self.vocab_size * self.hidden_dim
        
        # Transformer Layers
        # 每层: Self-Attn (4*H^2) + FFN (2*H*I)
        layer_params = 4 * self.hidden_dim * self.hidden_dim + \
                       2 * self.hidden_dim * self.intermediate_dim
        params['transformer'] = layer_params * self.num_layers
        
        # ECN (执行控制网络)
        params['ecn'] = self.ecn_layers * self.hidden_dim * self.hidden_dim
        
        # DMN (默认模式网络 - 联想记忆)
        params['dmn'] = self.dmn_associations * self.memory_dim * self.memory_dim
        
        # SN (显著性网络)
        params['sn'] = 2 * self.hidden_dim * self.sn_hidden_dim
        
        # Memory System
        params['memory'] = self.memory_slots * self.memory_dim
        
        # Vision Encoder (ViT风格)
        vision_params = self.vision_num_layers * \
                       (4 * self.vision_hidden_dim * self.vision_hidden_dim + \
                        2 * self.vision_hidden_dim * self.vision_hidden_dim * 4)
        params['vision'] = vision_params
        
        # Cross-Modal Fusion
        params['fusion'] = 3 * self.hidden_dim * self.fusion_dim
        
        # MLA压缩投影
        params['mla'] = 4 * self.hidden_dim * self.mla_latent_dim
        
        total = sum(params.values())
        return params, total
    
    def print_config(self):
        """打印配置详情"""
        params, total = self.calculate_params()
        
        print("=" * 70)
        print("NeuroFlow 1.5B Chinese+Code Model Configuration")
        print("=" * 70)
        
        print("\n[模型维度]")
        print(f"  Vocab Size:        {self.vocab_size:,} (中文:{self.chinese_vocab_size:,} + 代码:{self.code_vocab_size:,})")
        print(f"  Hidden Dim:        {self.hidden_dim}")
        print(f"  Intermediate Dim:  {self.intermediate_dim}")
        print(f"  Attention Heads:   {self.num_attention_heads}")
        print(f"  Head Dim:          {self.head_dim}")
        print(f"  Layers:            {self.num_layers}")
        
        print("\n[类脑网络]")
        print(f"  ECN Layers:        {self.ecn_layers}")
        print(f"  DMN Associations:  {self.dmn_associations}")
        print(f"  SN Hidden Dim:     {self.sn_hidden_dim}")
        
        print("\n[记忆系统]")
        print(f"  Memory Slots:      {self.memory_slots}")
        print(f"  Memory Dim:        {self.memory_dim}")
        print(f"  MLA Latent Dim:    {self.mla_latent_dim}")
        print(f"  MLA Memory Saving: {87.5}%")
        print(f"  Max Cache Len:     {self.max_cache_len}")
        
        print("\n[中文理解专精]")
        print(f"  Max Seq Len:       {self.max_seq_len}")
        print(f"  Position Encoding: {self.position_encoding}")
        print(f"  Tasks:             {len(self.chinese_tasks)}种")
        
        print("\n[编程能力专精]")
        print(f"  Languages:         {len(self.code_languages)}种")
        print(f"  Tasks:             {len(self.code_tasks)}种")
        
        print("\n[参数分布]")
        for name, p in params.items():
            print(f"  {name:14s}: {p:>12,} ({p/total*100:.1f}%)")
        
        print(f"\n  总参数量:        {total:>12,}")
        print(f"  目标参数量:      {1_500_000_000:>12,}")
        print(f"  差距:            {total - 1_500_000_000:>12,}")
        
        return params, total


class TrainingConfig:
    """训练配置"""
    
    # ==================== 基础训练 ====================
    epochs = 3
    batch_size = 16
    gradient_accumulation_steps = 4
    
    # ==================== 学习率 ====================
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    lr_scheduler = "cosine"
    
    # ==================== 优化器 ====================
    optimizer = "adamw_8bit"    # 8bit Adam节省内存
    max_grad_norm = 1.0
    
    # ==================== LoRA配置 ====================
    use_lora = True
    lora_r = 64                 # LoRA秩
    lora_alpha = 128            # LoRA alpha
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # ==================== 数据配置 ====================
    chinese_data_ratio = 0.6    # 中文数据占比
    code_data_ratio = 0.4       # 代码数据占比
    
    chinese_datasets = [
        "Skywork/Skywork-CN-corpus",
        "shibing624/multi-cn-nli",
        "Chinese-BERT/chinese-roberta-wwm-ext",
    ]
    
    code_datasets = [
        "bigcode/the-stack",
        "codeparrot/codeparrot-clean",
        "microsoft/codebert-base",
    ]
    
    # ==================== 硬件需求 ====================
    min_gpu_memory = "24GB"     # 最小GPU内存
    recommended_gpu = "A100 40GB / RTX 4090"
    
    # ==================== 保存配置 ====================
    output_dir = "./checkpoints/neuroflow-1.5b"
    save_strategy = "epoch"
    save_total_limit = 3
    
    # ==================== 评估 ====================
    eval_strategy = "epoch"
    eval_steps = 500
    
    # 中文评估任务
    chinese_eval_tasks = [
        "CMMLU",         # 中文多任务理解
        "C-Eval",        # 中文综合评估
        "CLUE",          # 中文语言理解
    ]
    
    # 编程评估任务
    code_eval_tasks = [
        "HumanEval",     # Python代码生成
        "MBPP",          # Python编程基准
        "CodeContests",  # 代码竞赛
    ]
    
    def print_config(self):
        """打印训练配置"""
        print("=" * 70)
        print("Training Configuration")
        print("=" * 70)
        
        print("\n[训练参数]")
        print(f"  Epochs:                    {self.epochs}")
        print(f"  Batch Size:                {self.batch_size}")
        print(f"  Gradient Accumulation:     {self.gradient_accumulation_steps}")
        print(f"  Effective Batch Size:      {self.batch_size * self.gradient_accumulation_steps}")
        
        print("\n[学习率]")
        print(f"  Learning Rate:             {self.learning_rate}")
        print(f"  Weight Decay:              {self.weight_decay}")
        print(f"  Warmup Ratio:              {self.warmup_ratio}")
        print(f"  Scheduler:                 {self.lr_scheduler}")
        
        print("\n[LoRA配置]")
        print(f"  LoRA r:                    {self.lora_r}")
        print(f"  LoRA alpha:                {self.lora_alpha}")
        print(f"  LoRA dropout:              {self.lora_dropout}")
        print(f"  Target Modules:            {len(self.lora_target_modules)}个")
        
        print("\n[数据比例]")
        print(f"  中文数据:                  {self.chinese_data_ratio*100}%")
        print(f"  代码数据:                  {self.code_data_ratio*100}%")
        
        print("\n[硬件需求]")
        print(f"  最小GPU内存:               {self.min_gpu_memory}")
        print(f"  推荐GPU:                   {self.recommended_gpu}")


if __name__ == "__main__":
    model_config = NeuroFlow1_5BConfig()
    model_config.print_config()
    
    print("\n")
    train_config = TrainingConfig()
    train_config.print_config()