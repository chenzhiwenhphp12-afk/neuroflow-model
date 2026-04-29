"""
NeuroFlow API在线训练脚本

使用方法:
    python api_train.py --api deepseek --key YOUR_API_KEY --task knowledge
    python api_train.py --api glm --key YOUR_API_KEY --task code
    python api_train.py --api deepseek --key YOUR_API_KEY --task reasoning

支持的API:
    - deepseek: DeepSeek API
    - glm: GLM-4 API (智谱AI)
    - openai: OpenAI API
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 直接导入api_training，避免torch依赖
sys.path.insert(0, str(Path(__file__).parent))

# 独立导入
import requests
import time

# API配置
class APIConfig:
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    OPENAI_BASE_URL = "https://api.openai.com/v1"

# BaseAPIClient
class BaseAPIClient:
    def __init__(self, api_key, base_url, model=None, timeout=60, max_retries=3):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        self.total_tokens = 0
        self.total_calls = 0
    
    def chat_completion(self, messages, model=None, temperature=0.7, max_tokens=2048, **kwargs):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    self.total_calls += 1
                    result = response.json()
                    if "usage" in result:
                        self.total_tokens += result["usage"].get("total_tokens", 0)
                    return result
                elif response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise Exception("Request timeout")
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise Exception(f"Request failed: {e}")
        
        raise Exception("Max retries exceeded")
    
    def get_stats(self):
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}

# DeepSeek客户端
class DeepSeekClient(BaseAPIClient):
    def __init__(self, api_key, model="deepseek-v4-flash", **kwargs):
        super().__init__(api_key, APIConfig.DEEPSEEK_BASE_URL, model, **kwargs)
    
    def reasoning_completion(self, messages, reasoning_effort="high", **kwargs):
        extra_body = {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort}
        return self.chat_completion(messages=messages, extra_body=extra_body, **kwargs)

# GLM客户端
class GLMClient(BaseAPIClient):
    def __init__(self, api_key, model="glm-4-flash", **kwargs):
        super().__init__(api_key, APIConfig.GLM_BASE_URL, model, **kwargs)

# APITrainer
class APITrainer:
    def __init__(self, api_client, output_dir="api_training_data"):
        self.client = api_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_data = []
    
    def generate_training_samples(self, topics, num_samples_per_topic=10, template="请解释{topic}的概念，并提供一个具体例子。"):
        samples = []
        for topic in topics:
            for i in range(num_samples_per_topic):
                prompt = template.format(topic=topic)
                messages = [
                    {"role": "system", "content": "你是一个知识丰富的AI助手。"},
                    {"role": "user", "content": prompt},
                ]
                try:
                    result = self.client.chat_completion(messages=messages)
                    if "choices" in result:
                        response = result["choices"][0]["message"]["content"]
                        samples.append({
                            "topic": topic,
                            "prompt": prompt,
                            "response": response,
                        })
                        print(f"Generated sample for '{topic}' ({i+1}/{num_samples_per_topic})")
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        self._save(samples, "generated_samples.json")
        return samples
    
    def knowledge_distillation(self, questions, batch_size=5):
        distilled = []
        for q in questions:
            messages = [{"role": "user", "content": q}]
            try:
                result = self.client.chat_completion(messages=messages, temperature=0.3)
                if "choices" in result:
                    answer = result["choices"][0]["message"]["content"]
                    distilled.append({"question": q, "answer": answer})
            except Exception as e:
                print(f"Error: {e}")
        
        self._save(distilled, "distilled_knowledge.json")
        return distilled
    
    def code_learning(self, tasks, language="python"):
        code_samples = []
        for task in tasks:
            prompt = f"请为以下任务生成{language}代码：\n任务: {task['description']}\n要求：代码清晰、高效，包含注释。"
            messages = [{"role": "user", "content": prompt}]
            try:
                result = self.client.chat_completion(messages=messages, max_tokens=4096)
                if "choices" in result:
                    code = result["choices"][0]["message"]["content"]
                    code_samples.append({"task": task, "code": code})
            except Exception as e:
                print(f"Error: {e}")
        
        self._save(code_samples, "learned_code.json")
        return code_samples
    
    def reasoning_training(self, problems, reasoning_effort="high"):
        reasoning_data = []
        for problem in problems:
            messages = [{"role": "user", "content": problem}]
            try:
                if hasattr(self.client, 'reasoning_completion'):
                    result = self.client.reasoning_completion(messages=messages, reasoning_effort=reasoning_effort)
                else:
                    result = self.client.chat_completion(messages=messages)
                
                if "choices" in result:
                    response = result["choices"][0]["message"]["content"]
                    reasoning_data.append({"problem": problem, "reasoning": response})
            except Exception as e:
                print(f"Error: {e}")
        
        self._save(reasoning_data, "reasoning_data.json")
        return reasoning_data
    
    def export_for_neuroflow(self):
        all_data = []
        
        for filename in ["generated_samples.json", "distilled_knowledge.json", "learned_code.json", "reasoning_data.json"]:
            filepath = self.output_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
        
        output_path = self.output_dir / "neuroflow_training_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(all_data)} samples to {output_path}")
        return str(output_path)
    
    def _save(self, data, filename):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {filepath}")


# 默认训练任务
DEFAULT_TOPICS = [
    "神经网络",
    "深度学习",
    "机器学习",
    "自然语言处理",
    "计算机视觉",
    "强化学习",
    "类脑计算",
    "记忆系统",
    "注意力机制",
    "知识蒸馏",
]

DEFAULT_CODE_TASKS = [
    {"type": "implementation", "description": "实现一个简单的神经网络前向传播"},
    {"type": "implementation", "description": "实现注意力机制的计算"},
    {"type": "implementation", "description": "实现记忆检索的注意力机制"},
    {"type": "optimization", "description": "优化矩阵乘法的性能"},
    {"type": "debugging", "description": "修复神经网络训练中的梯度消失问题"},
]

DEFAULT_REASONING_PROBLEMS = [
    "解释为什么类脑神经网络比传统神经网络更适合处理不确定性。",
    "分析记忆巩固在海马体到皮层迁移中的作用。",
    "比较ECN（执行控制网络）和DMN（默认模式网络）的功能差异。",
    "设计一个多模态融合方案，结合文本和图像特征。",
    "讨论如何实现模型的自我升级能力。",
]

DEFAULT_QUESTIONS = [
    "什么是类脑神经网络？",
    "如何实现长记忆机制？",
    "MLA KV压缩技术的原理是什么？",
    "INT8量化如何减少模型大小？",
    "类脑架构中的三大核心网络是什么？",
    "如何实现跨模态融合？",
    "什么是神经流形？",
    "LTP记忆巩固的原理是什么？",
    "如何优化模型推理速度？",
    "模型自我升级的实现方法？",
]


def parse_args():
    parser = argparse.ArgumentParser(description='NeuroFlow API Training')
    
    parser.add_argument('--api', type=str, required=True,
                        choices=['deepseek', 'glm', 'openai'],
                        help='API provider')
    
    parser.add_argument('--key', type=str, required=True,
                        help='API key')
    
    parser.add_argument('--task', type=str, default='knowledge',
                        choices=['knowledge', 'code', 'reasoning', 'distillation', 'test', 'full'],
                        help='Training task type')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default by provider)')
    
    parser.add_argument('--output', type=str, default='api_training_data',
                        help='Output directory')
    
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples per topic')
    
    parser.add_argument('--topics', type=str, nargs='+', default=None,
                        help='Custom topics')
    
    parser.add_argument('--thinking', type=bool, default=False,
                        help='Enable DeepSeek thinking mode')
    
    parser.add_argument('--effort', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort level')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path')
    
    return parser.parse_args()


def get_client(api: str, key: str, model: str = None):
    """获取API客户端"""
    if api == 'deepseek':
        default_model = "deepseek-v4-flash"
        return DeepSeekClient(api_key=key, model=model or default_model)
    elif api == 'glm':
        default_model = "glm-4-flash"
        return GLMClient(api_key=key, model=model or default_model)
    elif api == 'openai':
        default_model = "gpt-3.5-turbo"
        return BaseAPIClient(
            api_key=key,
            base_url=APIConfig.OPENAI_BASE_URL,
            model=model or default_model,
        )
    else:
        raise ValueError(f"Unknown API provider: {api}")


def run_knowledge_task(trainer: APITrainer, args):
    """知识生成任务"""
    print("\n=== Knowledge Generation Task ===")
    
    topics = args.topics or DEFAULT_TOPICS
    
    samples = trainer.generate_training_samples(
        topics=topics,
        num_samples_per_topic=args.samples,
    )
    
    print(f"Generated {len(samples)} knowledge samples")
    
    return samples


def run_code_task(trainer: APITrainer, args):
    """代码学习任务"""
    print("\n=== Code Learning Task ===")
    
    # 加载自定义任务或使用默认
    tasks = DEFAULT_CODE_TASKS
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            tasks = config.get('code_tasks', tasks)
    
    samples = trainer.code_learning(tasks=tasks, language="python")
    
    print(f"Generated {len(samples)} code samples")
    
    return samples


def run_reasoning_task(trainer: APITrainer, args):
    """推理训练任务"""
    print("\n=== Reasoning Training Task ===")
    
    problems = DEFAULT_REASONING_PROBLEMS
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            problems = config.get('reasoning_problems', problems)
    
    samples = trainer.reasoning_training(
        problems=problems,
        reasoning_effort=args.effort,
    )
    
    print(f"Generated {len(samples)} reasoning samples")
    
    return samples


def run_distillation_task(trainer: APITrainer, args):
    """知识蒸馏任务"""
    print("\n=== Knowledge Distillation Task ===")
    
    questions = DEFAULT_QUESTIONS
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            questions = config.get('questions', questions)
    
    samples = trainer.knowledge_distillation(questions=questions)
    
    print(f"Distilled {len(samples)} knowledge samples")
    
    return samples


def run_test_task(key: str, api: str):
    """API连接测试"""
    print("\n=== API Connection Test ===")
    
    client = get_client(api, key)
    
    try:
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )
        
        if "choices" in result:
            response = result["choices"][0]["message"]["content"]
            print(f"API connection successful!")
            print(f"Response: {response[:50]}...")
            return True
        else:
            print("API connection failed - no response")
            return False
    except Exception as e:
        print(f"API connection failed: {e}")
        return False


def run_full_training(trainer: APITrainer, args):
    """完整训练流程"""
    print("\n=== Full Training Pipeline ===")
    
    # 1. 测试连接
    print("\n[1] Testing API connection...")
    
    # 2. 知识生成
    print("\n[2] Generating knowledge samples...")
    knowledge_samples = run_knowledge_task(trainer, args)
    
    # 3. 代码学习
    print("\n[3] Learning code patterns...")
    code_samples = run_code_task(trainer, args)
    
    # 4. 推理训练
    print("\n[4] Training reasoning...")
    reasoning_samples = run_reasoning_task(trainer, args)
    
    # 5. 知识蒸馏
    print("\n[5] Distilling knowledge...")
    distilled_samples = run_distillation_task(trainer, args)
    
    # 6. 导出
    print("\n[6] Exporting for NeuroFlow...")
    output_path = trainer.export_for_neuroflow()
    
    # 统计
    stats = trainer.client.get_stats()
    
    print("\n=== Training Summary ===")
    print(f"Total API calls: {stats['total_calls']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Knowledge samples: {len(knowledge_samples)}")
    print(f"Code samples: {len(code_samples)}")
    print(f"Reasoning samples: {len(reasoning_samples)}")
    print(f"Distilled samples: {len(distilled_samples)}")
    print(f"Output file: {output_path}")
    
    return {
        "knowledge": knowledge_samples,
        "code": code_samples,
        "reasoning": reasoning_samples,
        "distilled": distilled_samples,
        "stats": stats,
        "output": output_path,
    }


def main():
    args = parse_args()
    
    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试任务
    if args.task == 'test':
        run_test_task(args.key, args.api)
        return
    
    # 创建客户端和训练器
    client = get_client(args.api, args.key, args.model)
    trainer = APITrainer(client, output_dir=str(output_dir))
    
    # 运行任务
    if args.task == 'knowledge':
        run_knowledge_task(trainer, args)
    elif args.task == 'code':
        run_code_task(trainer, args)
    elif args.task == 'reasoning':
        run_reasoning_task(trainer, args)
    elif args.task == 'distillation':
        run_distillation_task(trainer, args)
    elif args.task == 'full':
        result = run_full_training(trainer, args)
        
        # 保存完整结果
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                "total_samples": sum(len(v) for k, v in result.items() if isinstance(v, list)),
                "stats": result["stats"],
                "output": result["output"],
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
    
    # 最终导出
    if args.task != 'test':
        export_path = trainer.export_for_neuroflow()
        print(f"\nFinal export: {export_path}")
    
    # API统计
    stats = client.get_stats()
    print(f"\nAPI Statistics:")
    print(f"  Calls: {stats['total_calls']}")
    print(f"  Tokens: {stats['total_tokens']}")


if __name__ == '__main__':
    main()