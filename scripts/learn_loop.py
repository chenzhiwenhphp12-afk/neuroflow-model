#!/usr/bin/env python3
"""
NeuroFlow 全自动学习闭环系统
============================
循环周期: 每5小时完成一轮"学习→训练→评估→优化→准备"
运行方式: 从Hermes agent内调用 (需要web_search权限)
         execute_code 触发 → 全程自动完成

闭环流程图:
┌─────────────────────────────────────────────────────┐
│  ① 学习 (30s)  →  GitHub搜索 + KB文件生成 + HPC同步  │
│  ② 训练 (5h)   →  等待Daemon持续训练（已有后台运行）   │
│  ③ 评估 (10s)  →  采集recon/var/top5趋势 + 质量评分   │
│  ④ 优化 (20s)  →  自动调参建议 + 权重同步 + 策略调整  │
│  ⑤ 准备 (10s)  →  日志归档 + 计数器 + 初始化下一轮    │
└─────────────────────────────────────────────────────┘
"""

import os, sys, re, json, time, subprocess
from datetime import datetime

# ── 配置 ──
PROJECT_DIR = "/mnt/d/neuroflow-model"
KB_DIR = os.path.join(PROJECT_DIR, "knowledge_base")
HPC_HOST = "wuzh02.hpccube.com"
HPC_PORT = 65091
HPC_USER = "acxavb8ge5"
HPC_NF = "/work/home/acxavb8ge5/neuroflow-model"
HPC_KEY = os.path.expanduser("~/.ssh/hpc_key")

STATE_FILE = os.path.join(PROJECT_DIR, ".learn_loop_state.json")
LOG_FILE = "/tmp/neuroflow_learn_loop.log"

# ── 状态管理 ──
def load_state():
    """加载循环状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "cycle": 0,
        "started": datetime.now().isoformat(),
        "last_eval": None,
        "total_learned": 0,
        "best_recon": float('inf'),
        "best_var": 0,
        "best_fit": 0,
        "consecutive_improvements": 0,
        "optimization_history": [],
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── ① 学习阶段 ──
def phase_learn(state):
    """从GitHub搜索新知识 → 生成KB文件 → 同步HPC"""
    log("═" * 50)
    log("📚 阶段①: 学习")
    
    try:
        from hermes_tools import web_search, write_file
    except ImportError:
        log("❌ 不在Hermes上下文中，无法搜索")
        return 0
    
    # 搜索种子 (覆盖不同的技术领域)
    seeds = [
        "GitHub trending AI machine learning open source 2026",
        "GitHub security penetration testing vulnerability 2026",
        "GitHub systems programming Rust Zig compiler 2026",
        "GitHub database storage distributed system 2026",
        "GitHub web framework frontend Rust TypeScript 2026",
        "GitHub robotics embedded IoT edge computing 2026",
        "GitHub cryptography zero knowledge proof encryption 2026",
        "GitHub graphics rendering WebGPU Vulkan 3D 2026",
        "GitHub developer tool CLI terminal build system 2026",
        "GitHub networking proxy protocol service mesh 2026",
    ]
    
    new_count = 0
    for query in seeds:
        try:
            results = web_search(query=query, limit=3)
            web = results.get("data", {}).get("web", [])
            for item in web[:2]:
                title = item.get("title", "")
                desc = item.get("description", "")
                url = item.get("url", "")
                if not title or len(title) < 15:
                    continue
                
                # 生成KB文件
                num, fname = _next_kb_number()
                safe = re.sub(r'[^\w\-_]', '_', title)[:40]
                fname = f"{num:06d}_{safe}.txt"
                body = f"{title}: {desc}"[:2000]
                
                # 检测分类
                domain = _detect_domain(title, desc)
                content = f"[{domain}_自动学习|{title[:60]}] {body} [{url}]"
                
                write_file(os.path.join(KB_DIR, fname), content)
                new_count += 1
        except Exception as e:
            log(f"  ⚠️ 搜索失败: {e}")
    
    state["total_learned"] = state.get("total_learned", 0) + new_count
    log(f"  ✅ 新增 {new_count} 条知识 (累计 {state['total_learned']})")
    
    # 同步到HPC
    _sync_hpc(new_count)
    
    return new_count

def _next_kb_number():
    files = os.listdir(KB_DIR)
    nums = [int(f.split('_')[0]) for f in files if f.split('_')[0].isdigit()]
    return (max(nums) + 1 if nums else 840000), ""

def _detect_domain(title, desc):
    keywords = {
        "AI/ML": ["AI", "machine learning", "deep learning", "neural", "transformer", "LLM", "GPT", "agent", "model"],
        "安全": ["security", "vulnerability", "penetration", "hack", "crypto", "encryption", "attack", "malware"],
        "系统": ["system", "kernel", "compiler", "runtime", "virtual", "container", "OS", "linux", "driver"],
        "数据": ["database", "storage", "data", "SQL", "NoSQL", "query", "analytics", "stream"],
        "网络": ["network", "proxy", "protocol", "HTTP", "TCP", "load balance", "mesh", "CDN"],
        "前端/图形": ["frontend", "web", "UI", "graphics", "rendering", "WebGPU", "Vulkan", "CSS"],
        "工具": ["CLI", "terminal", "build", "deploy", "CI/CD", "automation", "devtool"],
        "硬件/嵌入": ["embedded", "IoT", "robot", "firmware", "hardware", "sensor", "edge"],
    }
    text = f"{title} {desc}".lower()
    for domain, kws in keywords.items():
        if any(kw in text for kw in kws):
            return domain
    return "其他"

def _sync_hpc(count):
    if count <= 0:
        return
    try:
        # SCP最新的KB文件到HPC
        cmd = ["scp", "-P", str(HPC_PORT), "-i", HPC_KEY,
               "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-q",
               os.path.join(KB_DIR, "*.txt"),
               f"{HPC_USER}@{HPC_HOST}:{HPC_NF}/knowledge_base/"]
        subprocess.run(cmd, capture_output=True, timeout=30)
        log(f"  📤 同步到HPC")
    except Exception as e:
        log(f"  ⚠️ HPC同步失败: {e}")


# ── ② 等待训练（已由后台Daemon自动完成）──
def phase_train_wait(state):
    """训练阶段: 检查daemon是否在运行"""
    log("═" * 50)
    log("🔄 阶段②: 训练 (已有后台Daemon)")
    
    # 检查daemon
    pid = None
    try:
        r = subprocess.run(["pgrep", "-f", "daemon_v3.py"], capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            pid = r.stdout.strip().split('\n')[0]
    except: pass
    
    if pid:
        log(f"  ✅ Daemon运行中 PID={pid}")
    else:
        log(f"  ⚠️ Daemon未运行，尝试启动...")
        try:
            subprocess.Popen(
                ["cd", PROJECT_DIR, "&&", 
                 "/home/administrator/.hermes/hermes-agent/venv/bin/python3", 
                 "daemon_v3.py", ">>", "daemon_v3.log", "2>&1"],
                cwd=PROJECT_DIR
            )
            log(f"  ✅ Daemon已启动")
        except Exception as e:
            log(f"  ❌ 启动失败: {e}")
    
    return bool(pid)


# ── ③ 评估阶段 ──
def phase_evaluate(state):
    """评估训练质量: 采集指标 → 计算评分 → 趋势分析"""
    log("═" * 50)
    log("📊 阶段③: 质量评估")
    
    log_path = os.path.join(PROJECT_DIR, "daemon_v3.log")
    if not os.path.exists(log_path):
        log("  ❌ 无日志文件")
        return None
    
    # 从日志提取最近的指标
    try:
        r = subprocess.run(["tail", "-50", log_path], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split('\n')
    except:
        lines = []
    
    metrics = {"recon": [], "var": [], "fit": [], "top5": [], "k": []}
    
    for line in lines:
        recon = re.search(r'recon=([0-9.]+)', line)
        var = re.search(r'var=([0-9.]+)', line)
        fit = re.search(r'fit=([0-9.]+)', line)
        k_val = re.search(r'k=(\d+)', line)
        top5 = re.search(r'top5.*?([0-9.]+)%', line)
        
        if recon: metrics["recon"].append(float(recon.group(1)))
        if var: metrics["var"].append(float(var.group(1)))
        if fit: metrics["fit"].append(float(fit.group(1)))
        if k_val: metrics["k"].append(int(k_val.group(1)))
        if top5: metrics["top5"].append(float(top5.group(1)))
    
    # 计算当前指标
    current = {}
    for key, vals in metrics.items():
        current[key] = vals[-1] if vals else 0
    
    # 趋势: 对比前5个值和后5个值
    trends = {}
    for key, vals in metrics.items():
        if len(vals) >= 10:
            first5 = sum(vals[:5]) / 5
            last5 = sum(vals[-5:]) / 5
            if key == "recon" or key == "var":
                # recon和var希望下降/上升
                if last5 < first5 and key == "recon":
                    trends[key] = "↑改善" if first5 > 0 else "稳定"
                elif last5 > first5 and key == "var":
                    trends[key] = "↑改善" 
                else:
                    trends[key] = "→稳定"
            else:
                diff_pct = (last5 - first5) / first5 * 100 if first5 > 0 else 0
                if abs(diff_pct) < 5:
                    trends[key] = "→稳定"
                elif diff_pct > 0:
                    trends[key] = f"↑+{diff_pct:.0f}%"
                else:
                    trends[key] = f"↓{diff_pct:.0f}%"
        else:
            trends[key] = "数据不足"
    
    # 质量评分 (0-100)
    score = 0
    if current.get("recon", 1) < 0.001:
        score += 30  # recon优秀
    elif current.get("recon", 1) < 0.005:
        score += 20
    else:
        score += 10
    
    var = current.get("var", 0)
    if var >= 0.0004:
        score += 30  # var优秀（有方差）
    elif var >= 0.0002:
        score += 20
    elif var > 0.0001:
        score += 10
    
    top5 = current.get("top5", 0)
    if top5 >= 22:
        score += 25
    elif top5 >= 20:
        score += 15
    else:
        score += 5
    
    fit = current.get("fit", 0)
    if fit >= 0.999:
        score += 15
    
    current["score"] = score
    current["trends"] = trends
    
    # 更新历史最佳
    if current.get("recon", 0) < state.get("best_recon", float('inf')) and current.get("recon", 0) > 0:
        state["best_recon"] = current["recon"]
        state["consecutive_improvements"] = state.get("consecutive_improvements", 0) + 1
    else:
        state["consecutive_improvements"] = 0
    
    if current.get("var", 0) > state.get("best_var", 0):
        state["best_var"] = current["var"]
    
    if current.get("fit", 0) > state.get("best_fit", 0):
        state["best_fit"] = current["fit"]
    
    # 报告
    log(f"  📈 recon={current.get('recon', '?'):.6f}  {trends.get('recon', '?')}")
    log(f"  📈 var={current.get('var', '?'):.4f}    {trends.get('var', '?')}")
    log(f"  📈 fit={current.get('fit', '?'):.4f}    {trends.get('fit', '?')}")
    log(f"  📈 top5={current.get('top5', '?'):.1f}%  {trends.get('top5', '?')}")
    log(f"  📊 质量评分: {score}/100")
    
    current["timestamp"] = datetime.now().isoformat()
    return current


# ── ④ 优化阶段 ──
def phase_optimize(state, evaluation):
    """基于评估结果，生成优化建议并执行"""
    log("═" * 50)
    log("🔧 阶段④: 优化")
    
    if not evaluation:
        log("  ⏭️ 无评估数据，跳过")
        return
    
    suggestions = []
    recon = evaluation.get("recon", 0.001)
    var = evaluation.get("var", 0)
    score = evaluation.get("score", 0)
    
    # 判断瓶颈
    if var < 0.0002 and recon < 0.001:
        # recon好但var差 → 需要注入方差
        suggestions.append("对比权重: 检查是否达到2.5上限, 若已封顶考虑增加记忆槽")
        suggestions.append("噪声注入: 如果对比已封顶, 增加输入噪声到0.3")
        suggestions.append("KB多样性: 需要更多样化的知识文件")
    elif var >= 0.0004 and recon >= 0.001:
        # var好但recon差 → 需要改善重建质量
        suggestions.append("学习率: 略微降低LR避免跳过最优解")
        suggestions.append("记忆库: 增加SAE的k_min到48以上, 保留更多特征")
    elif var < 0.0002 and recon >= 0.001:
        # 两者都差 → 架构问题
        suggestions.append("架构: 可能需要Leaky Integrator或GRU层")
        suggestions.append("维度: 检查HIDDEN_DIM/MEM_DIM是否过小")
    else:
        # 两者都好 → 微调
        if score < 70:
            suggestions.append("持续训练: 还需更多epoch成熟")
        if score >= 70:
            suggestions.append("✅ 系统健康, 保持当前配置")
    
    # 记录优化历史
    optimization = {
        "cycle": state["cycle"],
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "recon": recon,
        "var": var,
        "suggestions": suggestions,
    }
    state.setdefault("optimization_history", []).append(optimization)
    
    if suggestions:
        for s in suggestions:
            log(f"  💡 {s}")
    else:
        log(f"  ✅ 无需优化")
    
    return suggestions


# ── ⑤ 准备阶段 ──
def phase_prepare(state, evaluation):
    """日志归和 + 循环计数 + 权重同步到HPC"""
    log("═" * 50)
    log("🎯 阶段⑤: 下一轮准备")
    
    state["cycle"] += 1
    state["last_eval"] = datetime.now().isoformat()
    save_state(state)
    
    # 生成报告
    report = f"""
╔═══════════════════════════════════════════════════════╗
║  NeuroFlow 学习闭环 第{state['cycle']}轮完成                         ║
╠═══════════════════════════════════════════════════════╣
║  📊 质量评分: {evaluation.get('score', '?'):>2d}/100                  ║
║  📈 recon: {evaluation.get('recon', '?'):.6f}  var: {evaluation.get('var', '?'):.4f}  ║
║  📈 top5:  {evaluation.get('top5', '?'):.1f}%  fit: {evaluation.get('fit', '?'):.4f}    ║
║  📚 已学知识: {state.get('total_learned', 0)} 条 (累计)           ║
║  🏆 历史最佳 recon: {state.get('best_recon', '?'):.6f}              ║
║  🏆 历史最佳 var:   {state.get('best_var', '?'):.4f}                ║
╠═══════════════════════════════════════════════════════╣
║  磁盘: {_get_disk_usage()}                        ║
╚═══════════════════════════════════════════════════════╝"""
    log(report)
    
    # 同步权重到HPC (WSL→HPC)
    try:
        weights = os.path.join(PROJECT_DIR, "neuroflow_weights.npz")
        if os.path.exists(weights):
            cmd = ["scp", "-P", str(HPC_PORT), "-i", HPC_KEY,
                   "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-q",
                   weights,
                   f"{HPC_USER}@{HPC_HOST}:{HPC_NF}/neuroflow_weights.npz"]
            subprocess.run(cmd, capture_output=True, timeout=15)
            log(f"  🔄 权重同步到HPC")
    except Exception as e:
        log(f"  ⚠️ 权重同步失败: {e}")
    
    log(f"  ✅ 第{state['cycle']}轮完成, 等待下一轮")
    return report


def _get_disk_usage():
    try:
        r = subprocess.run(["df", "-h", PROJECT_DIR], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 4:
                return f"{parts[2]}/{parts[1]} ({parts[4]})"
    except: pass
    return "N/A"


# ── 主循环 ──
def main():
    log("=" * 55)
    log("🚀 NeuroFlow 全自动学习闭环 启动")
    
    state = load_state()
    
    # Phase 1: 学习
    new = phase_learn(state)
    
    # Phase 2: 训练 (检查daemon)
    daemon_ok = phase_train_wait(state)
    
    # Phase 3: 评估
    evaluation = phase_evaluate(state)
    
    # Phase 4: 优化
    phase_optimize(state, evaluation)
    
    # Phase 5: 准备
    report = phase_prepare(state, evaluation)
    
    log("=" * 55)
    print(report)
    print(f"\n日志: {LOG_FILE}")
    print(f"状态: {STATE_FILE}")
    
    return 0 if evaluation else 1


if __name__ == "__main__":
    sys.exit(main())
