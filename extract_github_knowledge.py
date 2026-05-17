#!/usr/bin/env python3
"""从 GitHub 仓库提取知识文本，生成结构化知识文件"""
import os, re, subprocess, sys, json

GITHUB_DIR = "/mnt/d/github_resources"
KB_DIR = "/mnt/d/neuroflow-model/knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)

# 获取现有文件数（用于文件编号起始）
existing = [f for f in os.listdir(KB_DIR) if f.endswith('.txt')]
# 获取最大编号
max_num = 0
for f in existing:
    try:
        num = int(f.split('_')[0])
        if num > max_num: max_num = num
    except: pass
file_counter = max_num + 1
print(f"现有知识文件: {len(existing)} 个, 起始编号: {file_counter}")

def clean_text(text: str) -> str:
    """清理文本：去HTML标签、多余空白、控制字符"""
    text = re.sub(r'<[^>]+>', ' ', text)  # HTML
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
    return text[:2000].strip()

def extract_markdown_content(text: str) -> list[str]:
    """从markdown提取段落（跳过代码块）"""
    paragraphs = []
    in_code = False
    current = []
    for line in text.split('\n'):
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        # 跳过标题标记、列表标记
        line_clean = re.sub(r'^#+\s*|^[*\-+]\s*|^\d+\.\s*', '', line).strip()
        if len(line_clean) > 30:
            current.append(line_clean)
        elif current and len(' '.join(current)) > 30:
            paragraphs.append(clean_text(' '.join(current)))
            current = []
    if current:
        paragraphs.append(clean_text(' '.join(current)))
    return [p for p in paragraphs if len(p) > 30]

def extract_python_knowledge(text: str) -> list[str]:
    """从Python文件提取知识（docstrings + 注释）"""
    items = []
    # docstrings
    for match in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        doc = clean_text(match.group(1))
        if len(doc) > 20:
            items.append(doc)
    # top-level comments (multi-line)
    for match in re.finditer(r'(?:^|\n)#(.+?)(?=\n\S|\Z)', text, re.DOTALL):
        comment = ' '.join(re.findall(r'#(.+)', match.group(0)))
        comment = clean_text(comment)
        if len(comment) > 20:
            items.append(comment)
    return items

def save_knowledge(content: str, source: str):
    """保存为知识文件"""
    global file_counter
    # 拆分成合理大小的段落
    paragraphs = []
    for para in content.split('\n\n'):
        p = clean_text(para)
        if len(p) > 30:
            paragraphs.append(p)
    
    for p in paragraphs[:3]:  # 每条最多3段
        fname = f"{file_counter:06d}_github_{source[:20].replace('/','_').replace(' ','_')}.txt"
        fpath = os.path.join(KB_DIR, fname)
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(p[:2000])
            file_counter += 1
        except:
            pass

# ═══════════════════════════════════════
# 遍历所有repo
# ═══════════════════════════════════════
total_saved = 0

for repo in sorted(os.listdir(GITHUB_DIR)):
    repo_path = os.path.join(GITHUB_DIR, repo)
    if not os.path.isdir(repo_path):
        continue
    
    print(f"\n📂 {repo}...", end=" ", flush=True)
    
    # 查找所有文档文件
    files = subprocess.run([
        'find', repo_path, '-type', 'f', 
        '(',
        '-name', '*.md', '-o', '-name', '*.txt', '-o',
        '-name', '*.rst', '-o', '-name', '*.html',
        ')'
    ], capture_output=True, text=True, timeout=30)
    
    doc_files = files.stdout.strip().split('\n') if files.stdout.strip() else []
    
    # 优先处理 README 和 docs 目录
    doc_files.sort(key=lambda f: (
        0 if 'readme' in f.lower() else 
        1 if '/docs/' in f.lower() else 
        2
    ))
    
    repo_count = 0
    for fpath in doc_files[:100]:  # 每个repo最多100个文件
        if not fpath:
            continue
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        if fpath.endswith('.md'):
            paras = extract_markdown_content(content)
        else:
            # 通用段落提取
            paras = [clean_text(p) for p in content.split('\n\n') if len(clean_text(p)) > 50]
        
        source_name = f"{repo}/{os.path.basename(fpath)}"
        for p in paras[:5]:  # 每个文件最多5段
            save_knowledge(p, source_name)
            repo_count += 1
    
    print(f"{repo_count} 条", end="", flush=True)
    total_saved += repo_count

# ═══════════════════════════════════════
# 从Python-100-Days提取更多代码知识
# ═══════════════════════════════════════
python_repo = os.path.join(GITHUB_DIR, "Python-100-Days")
if os.path.isdir(python_repo):
    print(f"\n\n🐍 Python-100-Days 代码知识提取...", end=" ", flush=True)
    py_files = subprocess.run([
        'find', python_repo, '-name', '*.py'
    ], capture_output=True, text=True, timeout=30)
    
    py_count = 0
    for fpath in py_files.stdout.strip().split('\n')[:200]:
        if not fpath: continue
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except: continue
        
        items = extract_python_knowledge(content)
        source_name = f"Python-100-Days/{os.path.basename(fpath)}"
        for item in items[:3]:
            save_knowledge(item, source_name)
            py_count += 1
    print(f"{py_count} 条", end="", flush=True)
    total_saved += py_count

# ═══════════════════════════════════════
# 批量合并策略：把短文本合并为长格式知识
# ═══════════════════════════════════════
print(f"\n\n📊 总计生成: {total_saved} 条新知识文件")
print(f"📁 知识库现在: {len(os.listdir(KB_DIR))} 个文件")

# 验证新文件
sample = sorted(os.listdir(KB_DIR), reverse=True)[:5]
print(f"📝 最新文件: {sample}")
