"""
NeuroFlow 优化效果测试

对比原始模型与优化模型：
1. 参数量
2. 内存占用
3. 推理速度
4. 准确率
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroflow.model import NeuroFlowModel
from neuroflow.model_optimized import OptimizedNeuroFlow, NeuroFlowLite


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """测量推理时间（毫秒）"""
    model.eval()
    x = torch.randn(*input_shape).to(device)
    
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # 计时
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # 转换为毫秒


def measure_memory(model: nn.Module, input_shape: tuple, device: torch.device) -> dict:
    """测量内存占用"""
    model.eval()
    x = torch.randn(*input_shape).to(device)
    
    # 参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 激活内存估算
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = model(x)
            activation_memory = torch.cuda.max_memory_allocated() - param_memory
        else:
            # CPU 模式下只能估算
            activation_memory = input_shape[0] * input_shape[1] * 4 * 10  # 粗略估算
    
    return {
        'params_bytes': param_memory,
        'params_mb': param_memory / 1024 / 1024,
        'activation_mb': activation_memory / 1024 / 1024 if isinstance(activation_memory, int) else activation_memory,
        'total_mb': param_memory / 1024 / 1024 + (activation_memory / 1024 / 1024 if isinstance(activation_memory, int) else 0),
    }


def test_accuracy(
    model: nn.Module,
    device: torch.device,
    num_samples: int = 1000,
    num_classes: int = 10,
) -> float:
    """测试分类准确率"""
    model.eval()
    input_dim = model.input_dim if hasattr(model, 'input_dim') else 512
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(num_samples // 32):
            # 生成随机输入和标签
            x = torch.randn(32, input_dim).to(device)
            labels = torch.randint(0, num_classes, (32,)).to(device)
            
            # 前向传播
            output = model(x)
            if isinstance(output, dict):
                output = output['output']
            
            # 计算准确率
            preds = output.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def run_benchmark():
    """运行完整基准测试"""
    print("=" * 60)
    print("NeuroFlow 优化基准测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 测试配置
    configs = [
        {'name': 'small', 'batch': 16, 'seq': 1, 'input_dim': 256},
        {'name': 'medium', 'batch': 32, 'seq': 1, 'input_dim': 512},
        {'name': 'large', 'batch': 64, 'seq': 1, 'input_dim': 1024},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"配置: {config['name']} (batch={config['batch']}, input_dim={config['input_dim']})")
        print("=" * 60)
        
        batch_size = config['batch']
        input_dim = config['input_dim']
        input_shape = (batch_size, input_dim)
        
        # 原始模型
        print("\n[原始 NeuroFlow]")
        try:
            original = NeuroFlowModel(
                input_dim=input_dim,
                hidden_dim=256,
                output_dim=10,
            ).to(device)
            
            original_params = count_parameters(original)
            original_time = measure_inference_time(original, input_shape, device)
            original_mem = measure_memory(original, input_shape, device)
            
            print(f"  参数量: {original_params:,} ({original_params/1e6:.2f}M)")
            print(f"  推理时间: {original_time:.2f} ms")
            print(f"  内存占用: {original_mem['params_mb']:.2f} MB (参数) + {original_mem['activation_mb']:.2f} MB (激活)")
            
            results[f"original_{config['name']}"] = {
                'params': original_params,
                'time_ms': original_time,
                'memory_mb': original_mem['total_mb'],
            }
        except Exception as e:
            print(f"  错误: {e}")
            results[f"original_{config['name']}"] = {'error': str(e)}
        
        # 优化模型
        print("\n[优化 NeuroFlow]")
        try:
            optimized = OptimizedNeuroFlow(
                input_dim=input_dim,
                hidden_dim=256,
                output_dim=10,
                use_quantization=False,  # 训练模式不用量化
            ).to(device)
            
            opt_params = count_parameters(optimized)
            opt_time = measure_inference_time(optimized, input_shape, device)
            opt_mem = measure_memory(optimized, input_shape, device)
            
            print(f"  参数量: {opt_params:,} ({opt_params/1e6:.2f}M)")
            print(f"  推理时间: {opt_time:.2f} ms")
            print(f"  内存占用: {opt_mem['params_mb']:.2f} MB (参数) + {opt_mem['activation_mb']:.2f} MB (激活)")
            
            # 计算提升
            if f"original_{config['name']}" in results and 'error' not in results[f"original_{config['name']}"]:
                orig = results[f"original_{config['name']}"]
                speedup = orig['time_ms'] / opt_time if opt_time > 0 else 0
                mem_save = (1 - opt_mem['total_mb'] / orig['memory_mb']) * 100 if orig['memory_mb'] > 0 else 0
                print(f"\n  优化提升:")
                print(f"    速度提升: {speedup:.2f}x")
                print(f"    内存节省: {mem_save:.1f}%")
            
            results[f"optimized_{config['name']}"] = {
                'params': opt_params,
                'time_ms': opt_time,
                'memory_mb': opt_mem['total_mb'],
            }
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results[f"optimized_{config['name']}"] = {'error': str(e)}
        
        # 轻量模型
        print("\n[轻量 NeuroFlow Lite]")
        try:
            lite = NeuroFlowLite(
                input_dim=input_dim,
            ).to(device)
            
            lite_params = count_parameters(lite)
            lite_time = measure_inference_time(lite, input_shape, device)
            lite_mem = measure_memory(lite, input_shape, device)
            
            print(f"  参数量: {lite_params:,} ({lite_params/1e6:.2f}M)")
            print(f"  推理时间: {lite_time:.2f} ms")
            print(f"  内存占用: {lite_mem['params_mb']:.2f} MB (参数) + {lite_mem['activation_mb']:.2f} MB (激活)")
            
            # 计算提升
            if f"original_{config['name']}" in results and 'error' not in results[f"original_{config['name']}"]:
                orig = results[f"original_{config['name']}"]
                speedup = orig['time_ms'] / lite_time if lite_time > 0 else 0
                mem_save = (1 - lite_mem['total_mb'] / orig['memory_mb']) * 100 if orig['memory_mb'] > 0 else 0
                param_reduce = (1 - lite_params / orig['params']) * 100 if orig['params'] > 0 else 0
                print(f"\n  相对原始模型:")
                print(f"    速度提升: {speedup:.2f}x")
                print(f"    参数减少: {param_reduce:.1f}%")
                print(f"    内存节省: {mem_save:.1f}%")
            
            results[f"lite_{config['name']}"] = {
                'params': lite_params,
                'time_ms': lite_time,
                'memory_mb': lite_mem['total_mb'],
            }
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results[f"lite_{config['name']}"] = {'error': str(e)}
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    # 计算平均优化效果
    avg_speedup_opt = 0
    avg_speedup_lite = 0
    avg_mem_save_opt = 0
    avg_mem_save_lite = 0
    count = 0
    
    for config in configs:
        key_orig = f"original_{config['name']}"
        key_opt = f"optimized_{config['name']}"
        key_lite = f"lite_{config['name']}"
        
        if key_orig in results and key_opt in results:
            if 'error' not in results[key_orig] and 'error' not in results[key_opt]:
                speedup = results[key_orig]['time_ms'] / results[key_opt]['time_ms']
                mem_save = (1 - results[key_opt]['memory_mb'] / results[key_orig]['memory_mb']) * 100
                avg_speedup_opt += speedup
                avg_mem_save_opt += mem_save
                count += 1
        
        if key_orig in results and key_lite in results:
            if 'error' not in results[key_orig] and 'error' not in results[key_lite]:
                speedup = results[key_orig]['time_ms'] / results[key_lite]['time_ms']
                mem_save = (1 - results[key_lite]['memory_mb'] / results[key_orig]['memory_mb']) * 100
                avg_speedup_lite += speedup
                avg_mem_save_lite += mem_save
    
    if count > 0:
        avg_speedup_opt /= count
        avg_mem_save_opt /= count
        avg_speedup_lite /= count
        avg_mem_save_lite /= count
        
        print(f"\n优化 NeuroFlow vs 原始:")
        print(f"  平均速度提升: {avg_speedup_opt:.2f}x")
        print(f"  平均内存节省: {avg_mem_save_opt:.1f}%")
        
        print(f"\nNeuroFlow Lite vs 原始:")
        print(f"  平均速度提升: {avg_speedup_lite:.2f}x")
        print(f"  平均内存节省: {avg_mem_save_lite:.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
    print("\n测试完成！")