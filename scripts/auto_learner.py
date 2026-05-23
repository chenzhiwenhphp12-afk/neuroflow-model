#!/usr/bin/env python3
"""
NeuroFlow 持续学习脚本 — 自动从GitHub发现、学习、投喂知识
============================================================
运行方式: python3 scripts/auto_learner.py
推荐: 放到cron每小时执行一次，或手动触发
依赖: Python 3.8+, requests (pip install requests)
"""

import os, sys, json, re, random, subprocess, time
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────
WSL_KB = "/mnt/d/neuroflow-model/knowledge_base"
HPC_HOST = "wuzh02.hpccube.com"
HPC_PORT = 65091
HPC_USER = "acxavb8ge5"
HPC_NF = "/work/home/acxavb8ge5/neuroflow-model"
HPC_KEY = os.path.expanduser("~/.ssh/hpc_key")

LOG_FILE = os.path.expanduser("~/.hermes/auto_learner.log")
TOPICS_DB = os.path.expanduser("~/.hermes/auto_learner_topics.json")

# 已学领域列表（避免重复）
LEARNED_TOPICS = set()
if os.path.exists(TOPICS_DB):
    with open(TOPICS_DB) as f:
        LEARNED_TOPICS = set(json.load(f))

# 搜索种子列表 — 每次随机选几个，确保多样性
SEEDS = [
    # AI / ML
    ("AI模型", "GitHub trending new AI model open source 2026"),
    ("AI Agent", "GitHub AI agent framework multi-agent 2026 trending"),
    ("机器学习", "GitHub machine learning library optimization framework 2026"),
    ("深度学习", "GitHub deep learning PyTorch TensorFlow training 2026 new"),
    ("自然语言", "GitHub NLP natural language processing LLM tokenizer 2026"),
    # 系统 / 底层
    ("系统编程", "GitHub systems programming Rust Zig kernel 2026 trending"),
    ("编译器", "GitHub compiler JIT LLVM MLIR optimization 2026"),
    ("操作系统", "GitHub operating system kernel embedded RTOS 2026"),
    ("虚拟化", "GitHub virtualization container KVM QEMU Firecracker 2026"),
    # 安全
    ("安全工具", "GitHub security pentesting vulnerability scanner 2026 trending"),
    ("密码学", "GitHub cryptography zero-knowledge proof encryption 2026"),
    ("隐私计算", "GitHub privacy compute federated learning differential privacy 2026"),
    # 基础设施
    ("数据库", "GitHub database engine SQL NoSQL vector database 2026"),
    ("网络", "GitHub networking proxy load balance service mesh 2026"),
    ("存储", "GitHub distributed storage file system object store 2026"),
    # 开发者工具
    ("开发者工具", "GitHub developer tool CLI terminal build system 2026"),
    ("可观测性", "GitHub observability monitoring tracing eBPF OpenTelemetry 2026"),
    ("CI/CD", "GitHub CI/CD pipeline deployment automation 2026"),
    # 新兴技术
    ("WebGPU", "GitHub WebGPU rendering graphics compute 2026"),
    ("WebAssembly", "GitHub WebAssembly WASM runtime emulator 2026"),
    ("区块链", "GitHub blockchain Web3 consensus DeFi 2026"),
    ("量子计算", "GitHub quantum computing simulator Qiskit Cirq 2026"),
    # 多模态与AI应用
    ("多模态AI", "GitHub multimodal vision language model VL MoE 2026"),
    ("语音AI", "GitHub TTS speech recognition voice cloning audio 2026"),
    ("计算机视觉", "GitHub computer vision object detection segmentation 2026"),
    ("具身智能", "GitHub embodied AI robotics manipulation control 2026"),
    # 数据/AI4S
    ("AI4S科学智能", "GitHub AI for science biology chemistry physics simulation 2026"),
    ("数据处理", "GitHub data pipeline ETL streaming analytics 2026"),
    # 边缘/嵌入式
    ("边缘计算", "GitHub edge computing IoT embedded TinyML 2026"),
    ("移动开发", "GitHub mobile development Flutter Kotlin Swift 2026"),
    # 游戏/图形
    ("游戏引擎", "GitHub game engine Godot Bevy Unity open source 2026"),
    ("计算机图形学", "GitHub graphics rendering ray tracing Vulkan 2026"),
    # Web
    ("Web框架", "GitHub web framework fullstack SSR React Rust 2026"),
    ("CSS/设计", "GitHub CSS design system UI component library 2026"),
    # 其它
    ("RAG检索", "GitHub RAG retrieval vector search knowledge graph 2026"),
    ("性能优化", "GitHub performance profiling optimization benchmark 2026"),
    ("自动化", "GitHub automation workflow script robot process 2026"),
    ("前端工程", "GitHub frontend build tool bundler Vite Turbopack 2026"),
]


def log(msg):
    """写日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def web_search(query):
    """使用Hermes web_search（必须从Hermes agent内调用）"""
    # 仅当在Hermes agent上下文中可用
    try:
        from hermes_tools import web_search as ws
        result = ws(query=query, limit=5)
        results = []
        for item in result.get("data", {}).get("web", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "desc": item.get("description", ""),
            })
        return results
    except ImportError:
        pass
    
    # 不在Hermes上下文中 → 返回空
    return []


def get_next_kb_number():
    """获取下一个KB文件编号"""
    files = os.listdir(WSL_KB)
    nums = []
    for f in files:
        try:
            nums.append(int(f.split("_")[0]))
        except (ValueError, IndexError):
            continue
    return max(nums) + 1 if nums else 840000


def topic_already_covered(domain, title):
    """检查是否已经学过类似主题"""
    key = f"{domain}:{title[:30]}"
    if key in LEARNED_TOPICS:
        return True
    # 模糊匹配: 如果领域相同且标题关键词相似，跳过
    for existing in LEARNED_TOPICS:
        existing_domain = existing.split(":")[0] if ":" in existing else ""
        if existing_domain == domain:
            existing_keywords = set(existing.split(":")[1].lower().split()[:5])
            new_keywords = set(title.lower().split()[:5])
            if len(existing_keywords & new_keywords) >= 3:
                return True
    return False


def write_knowledge_file(category, title, body):
    """写KB文件"""
    num = get_next_kb_number()
    safe_title = re.sub(r'[^\w\-_]', '_', title)[:40]
    fname = f"{num:06d}_{safe_title}.txt"
    fpath = os.path.join(WSL_KB, fname)
    
    # 截断body到2000字符
    if len(body) > 2000:
        body = body[:1997] + "..."
    
    content = f"[{category}_{title}|{title[:60]}] {body}"
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    
    # 记录已学
    key = f"{category}:{title[:30]}"
    LEARNED_TOPICS.add(key)
    with open(TOPICS_DB, "w", encoding="utf-8") as f:
        json.dump(list(LEARNED_TOPICS), f, ensure_ascii=False)
    
    log(f"  ✅ KB#{num:06d}: [{category}] {title}")
    return fname


def extract_top_results(results):
    """从搜索结果提取top N个有效结果"""
    valid = []
    for r in results:
        title = r.get("title", "")
        desc = r.get("desc", "")
        url = r.get("url", "")
        if title and len(title) > 10 and (desc or url):
            valid.append(r)
    return valid[:3]


def learn_topic(domain, query):
    """学习一个主题：搜索→提取→生成→保存"""
    if topic_already_covered(domain, query[:30]):
        return None
    
    log(f"🔍 学习 [{domain}]: {query}")
    results = web_search(query)
    if not results:
        log(f"  ⚠️ 无搜索结果")
        return None
    
    valid = extract_top_results(results)
    if not valid:
        return None
    
    # 生成知识正文
    body_parts = []
    for r in valid:
        t = r.get("title", "").replace("[", "(").replace("]", ")")
        d = r.get("desc", "").replace("[", "(").replace("]", ")")
        body_parts.append(f"{t}: {d}"[:400])
    
    body = " | ".join(body_parts)
    title = valid[0].get("title", query)[:60]
    
    return write_knowledge_file(domain, title, body)


def sync_to_hpc(files):
    """SCP同步到HPC"""
    if not files:
        return
    try:
        kb_dir = WSL_KB
        for f in files:
            cmd = [
                "scp", "-P", str(HPC_PORT), "-i", HPC_KEY,
                "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                "-q", f"{kb_dir}/{f}",
                f"{HPC_USER}@{HPC_HOST}:{HPC_NF}/knowledge_base/"
            ]
            subprocess.run(cmd, capture_output=True, timeout=15)
        log(f"  📤 同步{len(files)}个文件到HPC")
    except Exception as e:
        log(f"  ⚠️ HPC同步失败: {e}")


def check_daemon_status():
    """检查本地daemon状态"""
    try:
        result = subprocess.run(
            ["tail", "-1", "/mnt/d/neuroflow-model/daemon_v3.log"],
            capture_output=True, text=True, timeout=5
        )
        line = result.stdout.strip()
        if "recon" in line:
            # 提取recon和var
            recon = re.search(r'recon=([0-9.]+)', line)
            var = re.search(r'var=([0-9.]+)', line)
            batch = re.search(r'batch#(\d+)', line)
            return {
                "recon": recon.group(1) if recon else "?",
                "var": var.group(1) if var else "?",
                "batch": batch.group(1) if batch else "?",
            }
    except:
        pass
    return None


def generate_daily_report(new_count, daemon):
    """生成每日学习报告"""
    total = len(os.listdir(WSL_KB))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report = f"""
╔═══════════════════════════════════════════╗
║  NeuroFlow 学习报告  {now[:10]}          ║
╠═══════════════════════════════════════════╣
║  今日新增: {new_count:3d} 条                 ║
║  累计知识: {total:6d} 条                 ║
║  已学领域: {len(LEARNED_TOPICS):3d} 个               ║
"""
    if daemon:
        report += f"""║  Daemon:  batch#{daemon['batch']:>4s}              ║
║           recon={daemon['recon']}  var={daemon['var']}         ║
"""
    report += """╚═══════════════════════════════════════════╝"""
    return report


def main():
    """主循环"""
    log("=" * 55)
    log("🚀 NeuroFlow 自动学习启动")
    
    # 确保KB目录存在
    os.makedirs(WSL_KB, exist_ok=True)
    
    # 随机选3-5个种子（保证多样性）
    n_topics = min(5, max(3, len(SEEDS) // 8))
    selected = random.sample(SEEDS, min(n_topics, len(SEEDS)))
    
    new_files = []
    for domain, query in selected:
        try:
            result = learn_topic(domain, query)
            if result:
                new_files.append(result)
        except Exception as e:
            log(f"  ❌ 学习失败: {e}")
    
    # 同步到HPC
    if new_files:
        sync_to_hpc(new_files)
    
    # 生成报告
    daemon = check_daemon_status()
    report = generate_daily_report(len(new_files), daemon)
    log(report)
    
    # 输出摘要
    print(f"\n{'='*55}")
    print(f"学习完成: 新增{len(new_files)}条 / 累计{len(os.listdir(WSL_KB))}条")
    print(f"同步HPC: {'✅' if new_files else '⏭️ 无新增'}")
    print(f"日志: {LOG_FILE}")
    print(f"{'='*55}")
    
    return len(new_files)


if __name__ == "__main__":
    sys.exit(main())
