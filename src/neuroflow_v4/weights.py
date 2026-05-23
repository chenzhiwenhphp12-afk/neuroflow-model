"""
NeuroFlow v4 — 权重加载工具
==============================
支持从本地文件或 Hugging Face 自动下载。
"""

import os
import numpy as np
from typing import Dict, Optional

from . import config as C


def load_weights_from_npz(path: str) -> Dict[str, np.ndarray]:
    """从 .npz 文件加载权重
    
    Args:
        path: .npz 文件路径
    
    Returns:
        权重字典 (键名匹配 NeuroFlowV4 参数)
    """
    data = np.load(path, allow_pickle=True)
    return dict(data)


def get_default_cache_dir() -> str:
    """获取默认权重缓存目录 (~/.cache/neuroflow_v4/)"""
    cache = os.path.join(os.path.expanduser("~"), ".cache", "neuroflow_v4")
    os.makedirs(cache, exist_ok=True)
    return cache


def find_local_weights(search_paths: Optional[list] = None) -> Optional[str]:
    """在本地搜索权重文件
    
    Args:
        search_paths: 搜索路径列表, 默认:
          - ~/.cache/neuroflow_v4/neuroflow_weights_v4.npz
          - ./neuroflow_weights_v4.npz
          - ./neuroflow_weights.npz
          - 当前目录及上级目录
    
    Returns:
        找到的文件路径, 或 None
    """
    if search_paths is None:
        search_paths = [
            os.path.join(get_default_cache_dir(), C.WEIGHT_FILENAME),
            os.path.join(get_default_cache_dir(), "neuroflow_weights.npz"),
            os.path.join(".", C.WEIGHT_FILENAME),
            os.path.join(".", "neuroflow_weights.npz"),
            os.path.join("..", C.WEIGHT_FILENAME),
            os.path.join("..", "neuroflow_weights.npz"),
        ]
    
    for p in search_paths:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            return expanded
    
    return None


def download_from_huggingface(url: Optional[str] = None,
                              cache_dir: Optional[str] = None) -> str:
    """从 Hugging Face 下载权重文件
    
    Args:
        url: 下载 URL (默认使用 config 中的 HF_WEIGHT_URL)
        cache_dir: 缓存目录
    
    Returns:
        下载后的本地路径
    """
    import urllib.request
    
    if url is None:
        url = C.HF_WEIGHT_URL
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    dest = os.path.join(cache_dir, C.WEIGHT_FILENAME)
    
    if os.path.exists(dest):
        return dest
    
    print(f"📥 正在从 Hugging Face 下载模型权重...")
    print(f"   URL: {url}")
    print(f"   到: {dest}")
    
    tmp = dest + ".download"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"✅ 下载完成! ({size_mb:.1f} MB)")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"权重下载失败: {e}\n"
                           f"请手动下载: {url}\n"
                           f"然后放到: {dest}") from e
    
    return dest


def load_pretrained(local_path: Optional[str] = None,
                    auto_download: bool = True) -> Dict[str, np.ndarray]:
    """加载预训练权重 (自动查找本地 → 自动下载 HF)
    
    Args:
        local_path: 本地路径, 或 None 自动查找
        auto_download: 本地找不到时是否自动从 Hugging Face 下载
    
    Returns:
        权重字典
    """
    if local_path is None:
        local_path = find_local_weights()
    
    if local_path is not None:
        print(f"📂 从本地加载权重: {local_path}")
        return load_weights_from_npz(local_path)
    
    if auto_download:
        print(f"📂 本地未找到权重, 尝试从 Hugging Face 下载...")
        local_path = download_from_huggingface()
        return load_weights_from_npz(local_path)
    
    raise FileNotFoundError(
        f"未找到权重文件。\n"
        f"请从 {C.HF_WEIGHT_URL} 下载\n"
        f"或使用本地路径: {find_local_weights(search_paths=[]) or '.'}"
    )
