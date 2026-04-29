"""
权重导出工具

将训练好的模型权重导出为C++可读格式
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import struct


def export_weights_to_cpp(
    model: torch.nn.Module,
    output_dir: str,
    model_name: str = 'neuroflow',
    format: str = 'binary',
):
    """
    导出权重为C++格式
    
    Args:
        model: PyTorch模型
        output_dir: 输出目录
        model_name: 模型名称
        format: 格式 ('binary', 'npz', 'json')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights = {}
    shapes = {}
    
    # 提取权重
    for name, param in model.named_parameters():
        key = name.replace('.', '_')
        weights[key] = param.data.cpu().numpy()
        shapes[key] = list(param.shape)
        print(f"  {key}: {param.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    if format == 'binary':
        # 二进制格式（C++直接读取）
        bin_path = output_dir / f'{model_name}_weights.bin'
        
        with open(bin_path, 'wb') as f:
            # 写入元数据头
            # 1. 参数数量
            f.write(struct.pack('I', len(weights)))
            
            # 2. 每个参数的信息
            for key, arr in weights.items():
                # 名称长度 + 名称
                name_bytes = key.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                # 维度数量 + 维度
                f.write(struct.pack('I', len(arr.shape)))
                for dim in arr.shape:
                    f.write(struct.pack('I', dim))
                
                # 数据类型 (0=float32, 1=int8)
                dtype = 0 if arr.dtype == np.float32 else 1
                f.write(struct.pack('I', dtype))
                
                # 数据大小
                f.write(struct.pack('I', arr.size))
                
                # 数据
                if dtype == 0:
                    f.write(arr.astype(np.float32).tobytes())
                else:
                    f.write(arr.astype(np.int8).tobytes())
        
        print(f"Binary weights saved: {bin_path}")
        
        # 保存形状信息
        info_path = output_dir / f'{model_name}_weights_info.json'
        with open(info_path, 'w') as f:
            json.dump({
                'total_params': total_params,
                'weights': {k: {'shape': v, 'dtype': str(weights[k].dtype)} 
                           for k, v in shapes.items()},
            }, f, indent=2)
        print(f"Weight info saved: {info_path}")
    
    elif format == 'npz':
        # NPZ格式（numpy压缩）
        npz_path = output_dir / f'{model_name}_weights.npz'
        np.savez_compressed(npz_path, **weights)
        print(f"NPZ weights saved: {npz_path}")
        
        # 元数据
        meta_path = output_dir / f'{model_name}_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump({
                'total_params': total_params,
                'shapes': shapes,
            }, f, indent=2)
    
    elif format == 'json':
        # JSON格式（小模型适用）
        json_path = output_dir / f'{model_name}_weights.json'
        json_weights = {k: v.tolist() for k, v in weights.items()}
        with open(json_path, 'w') as f:
            json.dump(json_weights, f)
        print(f"JSON weights saved: {json_path}")
    
    return weights


def export_quantized_weights(
    model: torch.nn.Module,
    output_dir: str,
    model_name: str = 'neuroflow_lite',
):
    """
    导出量化权重（INT8）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    quantized_weights = {}
    scales = {}
    
    for name, param in model.named_parameters():
        key = name.replace('.', '_')
        data = param.data.cpu().numpy()
        
        # INT8量化
        max_val = np.abs(data).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        
        quantized = np.clip(data / scale, -128, 127).astype(np.int8)
        
        quantized_weights[key] = quantized
        scales[key] = scale
        
        print(f"  {key}: {param.shape} -> INT8 (scale={scale:.6f})")
    
    # 保存量化权重
    bin_path = output_dir / f'{model_name}_quantized.bin'
    
    with open(bin_path, 'wb') as f:
        # 头信息
        f.write(struct.pack('I', len(quantized_weights)))
        
        for key, arr in quantized_weights.items():
            # 名称
            name_bytes = key.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # 形状
            f.write(struct.pack('I', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('I', dim))
            
            # scale
            f.write(struct.pack('f', scales[key]))
            
            # 数据
            f.write(arr.tobytes())
    
    # 保存scale信息
    scale_path = output_dir / f'{model_name}_scales.json'
    with open(scale_path, 'w') as f:
        json.dump(scales, f, indent=2)
    
    original_size = sum(p.numel() * 4 for p in model.parameters())
    quantized_size = sum(p.numel() for p in model.parameters())
    
    print(f"\nQuantized weights saved: {bin_path}")
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Quantized size: {quantized_size / 1024:.2f} KB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    return quantized_weights, scales


def create_cpp_weight_loader(output_path: str):
    """
    创建C++权重加载器代码
    """
    code = '''// NeuroFlow 权重加载器
// Auto-generated by export_weights.py

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include "../tensor.hpp"

namespace neuroflow {

struct WeightInfo {
    std::string name;
    std::vector<size_t> shape;
    Tensor data;
};

class WeightLoader {
public:
    static std::unordered_map<std::string, Tensor> load_binary(const std::string& path) {
        std::unordered_map<std::string, Tensor> weights;
        
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open weight file: " + path);
        }
        
        // 读取参数数量
        uint32_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), 4);
        
        for (uint32_t i = 0; i < num_params; ++i) {
            // 名称
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), 4);
            std::string name(name_len, '\\0');
            file.read(&name[0], name_len);
            
            // 形状
            uint32_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), 4);
            std::vector<size_t> shape(num_dims);
            for (uint32_t d = 0; d < num_dims; ++d) {
                uint32_t dim;
                file.read(reinterpret_cast<char*>(&dim), 4);
                shape[d] = dim;
            }
            
            // 数据类型
            uint32_t dtype;
            file.read(reinterpret_cast<char*>(&dtype), 4);
            
            // 数据大小
            uint32_t data_size;
            file.read(reinterpret_cast<char*>(&data_size), 4);
            
            // 数据
            Tensor tensor(shape);
            if (dtype == 0) {
                std::vector<float> data(data_size);
                file.read(reinterpret_cast<char*>(data.data()), data_size * 4);
                tensor.from_float(data);
            } else {
                std::vector<int8_t> data(data_size);
                file.read(reinterpret_cast<char*>(data.data()), data_size);
                tensor.from_int8(data);
            }
            
            weights[name] = tensor;
        }
        
        file.close();
        return weights;
    }
    
    static std::unordered_map<std::string, Tensor> load_npz(const std::string& path) {
        // NPZ加载需要cnpy库或自定义实现
        // 这里简化为调用load_binary
        return load_binary(path.replace(path.find(".npz"), 4, ".bin"));
    }
};

} // namespace neuroflow
'''
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    print(f"C++ weight loader created: {output_path}")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    import argparse
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    from neuroflow.model import NeuroFlowModel
    
    parser = argparse.ArgumentParser(description='Export weights')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练检查点路径')
    parser.add_argument('--output_dir', type=str, default='weights')
    parser.add_argument('--format', type=str, default='binary',
                        choices=['binary', 'npz', 'json'])
    parser.add_argument('--quantize', type=bool, default=False,
                        help='是否量化为INT8')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 创建模型（需要用户根据配置）
    model = NeuroFlowModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 导出
    if args.quantize:
        export_quantized_weights(model, args.output_dir)
    else:
        export_weights_to_cpp(model, args.output_dir, format=args.format)
    
    # 创建C++加载器
    loader_path = Path(args.output_dir) / 'weight_loader.hpp'
    create_cpp_weight_loader(str(loader_path))