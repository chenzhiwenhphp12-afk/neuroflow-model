"""
NeuroFlow 1.5B 训练数据集配置
中文理解 + 编程能力 专精数据集
"""

from typing import List, Dict
import json

# ==================== 中文数据集 ====================

CHINESE_DATASETS = {
    # 通用中文语料
    "general": [
        {
            "name": "Skywork/Skywork-CN-corpus",
            "size": "100GB+",
            "description": "大规模中文语料库，包含网页、书籍、百科",
            "weight": 0.3,
        },
        {
            "name": "shibing624/multi-cn-nli",
            "size": "500K samples",
            "description": "中文自然语言推理数据集",
            "weight": 0.1,
        },
        {
            "name": "clue/clue-corpus2020",
            "size": "100GB+",
            "description": "CLUE中文语言理解基准语料",
            "weight": 0.2,
        },
    ],
    
    # 中文问答
    "qa": [
        {
            "name": "Hello-SimpleAI/HC3-Chinese",
            "size": "13K questions",
            "description": "中文问答数据集，涵盖多个领域",
            "weight": 0.1,
        },
        {
            "name": "squad-zh",
            "size": "10K passages",
            "description": "中文阅读理解SQUAD翻译版",
            "weight": 0.05,
        },
    ],
    
    # 中文技术文档
    "tech": [
        {
            "name": "Chinese-tech-docs",
            "size": "50GB",
            "description": "中文技术文档、API文档",
            "weight": 0.15,
        },
        {
            "name": "algorithm-cn",
            "size": "100K problems",
            "description": "中文算法题解数据集",
            "weight": 0.1,
        },
    ],
}

# ==================== 编程数据集 ====================

CODE_DATASETS = {
    # 通用代码数据集
    "general": [
        {
            "name": "bigcode/the-stack",
            "size": "300GB+",
            "description": "多语言代码数据集，涵盖358种语言",
            "weight": 0.3,
            "languages": ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript"],
        },
        {
            "name": "codeparrot/codeparrot-clean",
            "size": "50GB",
            "description": "清洗后的高质量代码数据",
            "weight": 0.2,
        },
    ],
    
    # Python专项
    "python": [
        {
            "name": "python-code-dataset",
            "size": "100GB",
            "description": "Python代码专项训练",
            "weight": 0.15,
        },
        {
            "name": "pypi-documentation",
            "size": "30GB",
            "description": "Python库文档和示例",
            "weight": 0.1,
        },
    ],
    
    # 代码解释/生成
    "tasks": [
        {
            "name": "HumanEval",
            "size": "164 problems",
            "description": "Python代码生成基准",
            "weight": 0.05,
        },
        {
            "name": "MBPP",
            "size": "974 problems",
            "description": "Python编程基准",
            "weight": 0.05,
        },
        {
            "name": "CodeAlpaca",
            "size": "20K examples",
            "description": "代码指令微调数据集",
            "weight": 0.1,
        },
    ],
}

# ==================== 中文编程混合数据集 ====================

CHINESE_CODE_DATASETS = {
    "leetcode-cn": {
        "name": "LeetCode-CN-Solutions",
        "size": "3000+ problems",
        "description": "力扣中文题解数据集",
        "weight": 0.3,
    },
    "csdn-blogs": {
        "name": "CSDN-Code-Blogs",
        "size": "50GB",
        "description": "CSDN编程博客数据",
        "weight": 0.2,
    },
    "zhihu-tech": {
        "name": "Zhihu-Tech-Answers",
        "size": "30GB",
        "description": "知乎技术问答数据",
        "weight": 0.15,
    },
    "github-cn": {
        "name": "GitHub-CN-Projects",
        "size": "100GB",
        "description": "中文GitHub项目代码",
        "weight": 0.35,
    },
}

# ==================== 数据集加载配置 ====================

class DatasetLoaderConfig:
    """数据集加载配置"""
    
    def __init__(self):
        # 中文数据比例
        self.chinese_ratio = 0.6
        
        # 代码数据比例
        self.code_ratio = 0.4
        
        # 中文代码混合数据比例
        self.chinese_code_ratio = 0.3
        
        # 最大序列长度
        self.max_seq_length = 8192
        
        # 批处理大小
        self.batch_size = 16
        
        # 数据预处理
        self.preprocessing = {
            "tokenizer": "bpe",
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        
        # 数据增强
        self.augmentation = {
            "code_augmentation": True,      # 代码格式变换
            "chinese_augmentation": True,   # 中文表达变换
            "translation_pair": True,       # 中英文对照
        }
    
    def get_total_weight(self):
        """计算各类数据权重"""
        weights = {
            "chinese_general": self.chinese_ratio * 0.5,
            "chinese_qa": self.chinese_ratio * 0.2,
            "chinese_tech": self.chinese_ratio * 0.3,
            "code_general": self.code_ratio * 0.5,
            "code_python": self.code_ratio * 0.3,
            "code_tasks": self.code_ratio * 0.2,
            "chinese_code": self.chinese_code_ratio,
        }
        return weights


# ==================== 数据质量筛选 ====================

QUALITY_FILTERS = {
    "min_length": 100,           # 最小文本长度
    "max_length": 8192,          # 最大文本长度
    "min_code_lines": 5,         # 代码最小行数
    "require_comments": True,    # 代码必须有注释
    "chinese_ratio_min": 0.3,    # 中文内容最小比例
    "language_filter": ["zh", "en", "code"],
    "remove_duplicates": True,   # 去重
    "quality_score_min": 0.7,    # 最小质量分数
}


# ==================== 数据加载脚本示例 ====================

def load_chinese_datasets():
    """加载中文数据集"""
    from datasets import load_dataset
    
    datasets_list = []
    
    # 加载Skywork中文语料
    try:
        skywork = load_dataset("Skywork/Skywork-CN-corpus", split="train")
        datasets_list.append(("skywork", skywork, 0.3))
    except Exception as e:
        print(f"Skywork加载失败: {e}")
    
    # 加载中文NLI
    try:
        nli = load_dataset("shibing624/multi-cn-nli", split="train")
        datasets_list.append(("nli", nli, 0.1))
    except Exception as e:
        print(f"NLI加载失败: {e}")
    
    return datasets_list


def load_code_datasets():
    """加载编程数据集"""
    from datasets import load_dataset
    
    datasets_list = []
    
    # 加载The Stack
    try:
        stack = load_dataset("bigcode/the-stack", 
                             data_dir="data/python",
                             split="train")
        datasets_list.append(("stack", stack, 0.3))
    except Exception as e:
        print(f"The Stack加载失败: {e}")
    
    # 加载HumanEval
    try:
        humaneval = load_dataset("openai/humaneval", split="test")
        datasets_list.append(("humaneval", humaneval, 0.05))
    except Exception as e:
        print(f"HumanEval加载失败: {e}")
    
    return datasets_list


def prepare_training_data():
    """准备训练数据"""
    
    print("=" * 60)
    print("NeuroFlow 1.5B 数据集配置")
    print("=" * 60)
    
    config = DatasetLoaderConfig()
    weights = config.get_total_weight()
    
    print("\n[数据权重分配]")
    for name, weight in weights.items():
        print(f"  {name:20s}: {weight*100:.1f}%")
    
    print("\n[中文数据集]")
    for category, datasets in CHINESE_DATASETS.items():
        print(f"\n  {category}:")
        for ds in datasets:
            print(f"    - {ds['name']:30s} ({ds['size']}) weight={ds['weight']}")
    
    print("\n[编程数据集]")
    for category, datasets in CODE_DATASETS.items():
        print(f"\n  {category}:")
        for ds in datasets:
            print(f"    - {ds['name']:30s} ({ds['size']}) weight={ds['weight']}")
    
    print("\n[中文编程混合数据集]")
    for name, ds in CHINESE_CODE_DATASETS.items():
        print(f"  - {ds['name']:30s} ({ds['size']}) weight={ds['weight']}")
    
    print("\n[质量筛选标准]")
    for key, value in QUALITY_FILTERS.items():
        print(f"  {key:20s}: {value}")
    
    total_weight = sum(weights.values())
    print(f"\n  总权重: {total_weight:.2f}")
    
    return config


if __name__ == "__main__":
    prepare_training_data()