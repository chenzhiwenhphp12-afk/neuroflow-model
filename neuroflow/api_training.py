"""
NeuroFlow API在线训练模块

支持多种API：
- DeepSeek API (https://api.deepseek.com)
- GLM-4 API (智谱AI开放平台)
- OpenAI API (兼容格式)
- 其他兼容API

功能：
1. 模型知识注入 - 通过API生成训练数据
2. 代码能力学习 - 学习GLM编码能力
3. 知识蒸馏 - 从大模型蒸馏到NeuroFlow
4. 在线推理 - 使用API增强推理能力
"""

import requests
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import hashlib


class APIConfig:
    """API配置"""
    
    # DeepSeek API
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]
    
    # GLM-4 API (智谱)
    GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    GLM_MODELS = ["glm-4", "glm-4-flash", "glm-4-air", "glm-4-airx", "glm-3-turbo"]
    
    # OpenAI API
    OPENAI_BASE_URL = "https://api.openai.com/v1"
    OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    # 自定义API
    CUSTOM_BASE_URL = None


class BaseAPIClient:
    """API客户端基类"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
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
        
        # 统计
        self.total_tokens = 0
        self.total_calls = 0
        self.total_cost = 0.0
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict,
        stream: bool = False,
    ) -> Dict:
        """发送请求"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
                
                if response.status_code == 200:
                    self.total_calls += 1
                    return response.json() if not stream else response
                elif response.status_code == 429:  # Rate limit
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
    
    def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs,
    ) -> Dict:
        """聊天补全"""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        payload.update(kwargs)
        
        result = self._make_request("chat/completions", payload, stream)
        
        if not stream and "usage" in result:
            self.total_tokens += result["usage"].get("total_tokens", 0)
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


class DeepSeekClient(BaseAPIClient):
    """DeepSeek API客户端"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            base_url=APIConfig.DEEPSEEK_BASE_URL,
            model=model,
            **kwargs,
        )
    
    def chat_completion(
        self,
        messages: List[Dict],
        thinking: bool = False,
        reasoning_effort: str = "medium",  # low, medium, high
        **kwargs,
    ) -> Dict:
        """DeepSeek聊天补全（支持思考模式）"""
        extra_body = {}
        if thinking:
            extra_body["thinking"] = {"type": "enabled"}
            extra_body["reasoning_effort"] = reasoning_effort
        
        return super().chat_completion(
            messages=messages,
            extra_body=extra_body,
            **kwargs,
        )
    
    def reasoning_completion(
        self,
        messages: List[Dict],
        reasoning_effort: str = "high",
        **kwargs,
    ) -> Dict:
        """推理模式补全"""
        return self.chat_completion(
            messages=messages,
            thinking=True,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )


class GLMClient(BaseAPIClient):
    """GLM-4 API客户端（智谱AI）"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            base_url=APIConfig.GLM_BASE_URL,
            model=model,
            **kwargs,
        )
    
    def chat_completion(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Dict:
        """GLM聊天补全（支持工具调用）"""
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        payload.update(kwargs)
        
        return self._make_request("chat/completions", payload)
    
    def code_completion(
        self,
        prompt: str,
        language: str = "python",
        **kwargs,
    ) -> str:
        """代码补全"""
        messages = [
            {"role": "system", "content": f"你是一个{language}编程专家。请生成高质量的代码。"},
            {"role": "user", "content": prompt},
        ]
        
        result = self.chat_completion(messages=messages, **kwargs)
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        
        return ""


class APITrainer:
    """API在线训练器"""
    
    def __init__(
        self,
        api_client: BaseAPIClient,
        output_dir: str = "api_training_data",
    ):
        self.client = api_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练数据缓存
        self.training_data = []
        self.knowledge_base = {}
    
    def generate_training_samples(
        self,
        topics: List[str],
        num_samples_per_topic: int = 10,
        template: str = "请解释{topic}的概念，并提供一个具体例子。",
    ) -> List[Dict]:
        """生成训练样本"""
        samples = []
        
        for topic in topics:
            for i in range(num_samples_per_topic):
                prompt = template.format(topic=topic)
                
                messages = [
                    {"role": "system", "content": "你是一个知识丰富的AI助手，请提供准确、详细的回答。"},
                    {"role": "user", "content": prompt},
                ]
                
                try:
                    result = self.client.chat_completion(messages=messages)
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        response = result["choices"][0]["message"]["content"]
                        
                        sample = {
                            "topic": topic,
                            "prompt": prompt,
                            "response": response,
                            "model": self.client.model,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        samples.append(sample)
                        
                        print(f"Generated sample for '{topic}' ({i+1}/{num_samples_per_topic})")
                        
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    continue
        
        self.training_data.extend(samples)
        self._save_training_data(samples, "generated_samples.json")
        
        return samples
    
    def knowledge_distillation(
        self,
        questions: List[str],
        batch_size: int = 5,
    ) -> List[Dict]:
        """知识蒸馏 - 从大模型获取知识"""
        distilled = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            
            for q in batch:
                messages = [
                    {"role": "system", "content": "请用简洁、准确的语言回答问题。"},
                    {"role": "user", "content": q},
                ]
                
                try:
                    result = self.client.chat_completion(
                        messages=messages,
                        temperature=0.3,  # 低温度，更确定性的输出
                    )
                    
                    if "choices" in result:
                        answer = result["choices"][0]["message"]["content"]
                        
                        distilled.append({
                            "question": q,
                            "answer": answer,
                            "tokens": result.get("usage", {}).get("total_tokens", 0),
                        })
                        
                except Exception as e:
                    print(f"Distillation error: {e}")
                    continue
        
        self._save_training_data(distilled, "distilled_knowledge.json")
        
        return distilled
    
    def code_learning(
        self,
        tasks: List[Dict],
        language: str = "python",
    ) -> List[Dict]:
        """代码能力学习"""
        code_samples = []
        
        for task in tasks:
            task_type = task.get("type", "implementation")
            description = task.get("description", "")
            
            prompt = f"""
请为以下任务生成{language}代码：

任务类型: {task_type}
描述: {description}

要求：
1. 代码清晰、高效
2. 添加适当的注释
3. 包含错误处理
4. 提供使用示例
"""
            
            messages = [
                {"role": "system", "content": f"你是一个{language}编程专家。"},
                {"role": "user", "content": prompt},
            ]
            
            try:
                result = self.client.chat_completion(messages=messages, max_tokens=4096)
                
                if "choices" in result:
                    code = result["choices"][0]["message"]["content"]
                    
                    code_samples.append({
                        "task": task,
                        "language": language,
                        "code": code,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    
            except Exception as e:
                print(f"Code learning error: {e}")
                continue
        
        self._save_training_data(code_samples, "learned_code.json")
        
        return code_samples
    
    def reasoning_training(
        self,
        problems: List[str],
        reasoning_effort: str = "high",
    ) -> List[Dict]:
        """推理能力训练（DeepSeek思考模式）"""
        reasoning_data = []
        
        for problem in problems:
            messages = [
                {"role": "system", "content": "请仔细分析问题，展示完整的推理过程。"},
                {"role": "user", "content": problem},
            ]
            
            try:
                # 使用思考模式
                if hasattr(self.client, 'reasoning_completion'):
                    result = self.client.reasoning_completion(
                        messages=messages,
                        reasoning_effort=reasoning_effort,
                    )
                else:
                    result = self.client.chat_completion(messages=messages)
                
                if "choices" in result:
                    response = result["choices"][0]["message"]["content"]
                    
                    reasoning_data.append({
                        "problem": problem,
                        "reasoning": response,
                        "effort": reasoning_effort,
                    })
                    
            except Exception as e:
                print(f"Reasoning training error: {e}")
                continue
        
        self._save_training_data(reasoning_data, "reasoning_data.json")
        
        return reasoning_data
    
    def multi_turn_conversation(
        self,
        scenarios: List[Dict],
    ) -> List[Dict]:
        """多轮对话训练"""
        conversations = []
        
        for scenario in scenarios:
            context = scenario.get("context", "")
            turns = scenario.get("turns", [])
            
            messages = [
                {"role": "system", "content": context},
            ]
            
            conversation_history = []
            
            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                
                messages.append({"role": role, "content": content})
                
                if role == "user":
                    try:
                        result = self.client.chat_completion(messages=messages)
                        
                        if "choices" in result:
                            response = result["choices"][0]["message"]["content"]
                            
                            messages.append({"role": "assistant", "content": response})
                            
                            conversation_history.append({
                                "user": content,
                                "assistant": response,
                            })
                            
                    except Exception as e:
                        print(f"Conversation error: {e}")
                        continue
            
            conversations.append({
                "scenario": scenario,
                "history": conversation_history,
            })
        
        self._save_training_data(conversations, "conversations.json")
        
        return conversations
    
    def _save_training_data(self, data: List[Dict], filename: str):
        """保存训练数据"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {filepath}")
    
    def export_for_neuroflow(self, output_format: str = "json") -> str:
        """导出为NeuroFlow训练格式"""
        all_data = {
            "generated": [],
            "distilled": [],
            "code": [],
            "reasoning": [],
            "conversations": [],
        }
        
        # 加载所有数据
        for key in all_data:
            filepath = self.output_dir / f"{key}_{'knowledge' if key == 'distilled' else 'samples' if key == 'generated' else 'code' if key == 'code' else 'data' if key == 'reasoning' else 'conversations'}.json"
            alt_filepath = self.output_dir / f"{'generated_samples' if key == 'generated' else 'distilled_knowledge' if key == 'distilled' else 'learned_code' if key == 'code' else 'reasoning_data' if key == 'reasoning' else 'conversations'}.json"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_data[key] = json.load(f)
            elif alt_filepath.exists():
                with open(alt_filepath, 'r', encoding='utf-8') as f:
                    all_data[key] = json.load(f)
        
        # 合并并格式化
        neuroflow_data = []
        
        for key, items in all_data.items():
            for item in items:
                if key == "generated":
                    neuroflow_data.append({
                        "input": item["prompt"],
                        "output": item["response"],
                        "type": "qa",
                        "topic": item["topic"],
                    })
                elif key == "distilled":
                    neuroflow_data.append({
                        "input": item["question"],
                        "output": item["answer"],
                        "type": "distillation",
                    })
                elif key == "code":
                    neuroflow_data.append({
                        "input": item["task"]["description"],
                        "output": item["code"],
                        "type": "code",
                        "language": item["language"],
                    })
                elif key == "reasoning":
                    neuroflow_data.append({
                        "input": item["problem"],
                        "output": item["reasoning"],
                        "type": "reasoning",
                    })
        
        # 保存
        output_path = self.output_dir / "neuroflow_training_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(neuroflow_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(neuroflow_data)} samples to {output_path}")
        
        return str(output_path)


class NeuroFlowAPIIntegration:
    """NeuroFlow与API集成"""
    
    def __init__(
        self,
        neuroflow_model_path: str,
        api_client: BaseAPIClient,
    ):
        self.model_path = neuroflow_model_path
        self.api_client = api_client
        
    def enhance_reasoning(
        self,
        input_text: str,
        use_api: bool = True,
    ) -> Dict:
        """增强推理能力"""
        result = {
            "input": input_text,
            "local_reasoning": None,
            "api_reasoning": None,
            "combined": None,
        }
        
        # 本地推理（如果有模型）
        # result["local_reasoning"] = self._local_forward(input_text)
        
        # API推理
        if use_api:
            messages = [
                {"role": "system", "content": "请分析并回答以下问题。"},
                {"role": "user", "content": input_text},
            ]
            
            try:
                api_result = self.api_client.chat_completion(messages=messages)
                if "choices" in api_result:
                    result["api_reasoning"] = api_result["choices"][0]["message"]["content"]
            except Exception as e:
                result["api_error"] = str(e)
        
        return result
    
    def adaptive_learning(
        self,
        feedback: str,
        context: str,
    ) -> Dict:
        """自适应学习 - 根据反馈改进"""
        messages = [
            {"role": "system", "content": "你是一个学习助手，根据反馈改进知识。"},
            {"role": "user", "content": f"上下文: {context}\n反馈: {feedback}\n请提供改进后的回答。"},
        ]
        
        result = self.api_client.chat_completion(messages=messages)
        
        return result


# ==================== 使用示例 ====================

def create_deepseek_trainer(api_key: str, model: str = "deepseek-v4-flash") -> APITrainer:
    """创建DeepSeek训练器"""
    client = DeepSeekClient(api_key=api_key, model=model)
    trainer = APITrainer(client)
    return trainer


def create_glm_trainer(api_key: str, model: str = "glm-4-flash") -> APITrainer:
    """创建GLM训练器"""
    client = GLMClient(api_key=api_key, model=model)
    trainer = APITrainer(client)
    return trainer


def test_api_connection(api_key: str, provider: str = "deepseek") -> bool:
    """测试API连接"""
    try:
        if provider == "deepseek":
            client = DeepSeekClient(api_key=api_key)
            result = client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
        elif provider == "glm":
            client = GLMClient(api_key=api_key)
            result = client.chat_completion(
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10,
            )
        
        if "choices" in result:
            print(f"API连接成功! Provider: {provider}")
            print(f"Response: {result['choices'][0]['message']['content'][:50]}...")
            return True
        
    except Exception as e:
        print(f"API连接失败: {e}")
        return False
    
    return False