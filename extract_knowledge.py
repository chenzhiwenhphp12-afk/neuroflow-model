"""从现有 GitHub 仓库提取所有文本知识，直接写入 knowledge_base/"""
import os, re, sys
from pathlib import Path

KB_DIR = "/mnt/d/neuroflow-model/knowledge_base"
REPOS_DIR = "/mnt/d/github_resources"

def extract_text_from_md(content: str) -> str:
    """提取markdown中的纯文本，跳过代码块、URL、导航等"""
    # 移除以 ``` 包围的代码块
    content = re.sub(r'```[\s\S]*?```', '', content)
    # 移除行内代码
    content = re.sub(r'`[^`]+`', '', content)
    # 移除图片
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 移除链接（但保留链接文本）
    content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
    # 移除HTML注释
    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    # 移除markdown标题标记
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # 移除列表标记
    content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
    # 移除数字列表
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
    # 移除分隔线
    content = re.sub(r'^-{3,}$', '', content, flags=re.MULTILINE)
    # 移除多余空白
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'  +', ' ', content)
    content = content.strip()
    return content

def extract_text_from_txt(content: str) -> str:
    """从纯文本提取知识"""
    lines = content.split('\n')
    # 跳过太短的行
    meaningful = [l.strip() for l in lines if len(l.strip()) > 30]
    return '\n'.join(meaningful)

def process_file(filepath: str) -> list[str]:
    """处理一个文件，返回知识文本列表"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.md':
        text = extract_text_from_md(content)
    elif ext == '.txt':
        text = extract_text_from_txt(content)
    else:
        return []
    
    if len(text) < 200:
        return []
    
    # 按段落拆分，每段一条知识
    paragraphs = []
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if len(para) >= 150:
            paragraphs.append(para[:5000])  # 限制长度
    
    return paragraphs

def get_file_id():
    """获取下一个可用文件ID"""
    files = sorted(Path(KB_DIR).glob('*.txt'))
    if not files:
        return 129884
    # 找到最大的数字前缀
    max_id = 0
    for f in files:
        stem = f.stem
        try:
            id_val = int(stem.split('_')[0])
            if id_val > max_id:
                max_id = id_val
        except ValueError:
            continue
    return max_id + 1 if max_id > 0 else 129884

def main():
    nid = get_file_id()
    total_created = 0
    total_failed = 0
    
    # 扫描所有仓库
    for repo_dir in sorted(os.listdir(REPOS_DIR)):
        repo_path = os.path.join(REPOS_DIR, repo_dir)
        if not os.path.isdir(repo_path):
            continue
        
        # 跳过 .git 目录
        if repo_dir.startswith('.'):
            continue
        
        repo_count = 0
        for root, dirs, files in os.walk(repo_path):
            # 跳过 .git 目录
            dirs[:] = [d for d in dirs if d != '.git']
            # 跳过 node_modules
            dirs[:] = [d for d in dirs if d != 'node_modules']
            
            for fname in files:
                if not fname.endswith(('.md', '.txt')):
                    continue
                
                fpath = os.path.join(root, fname)
                texts = process_file(fpath)
                
                for text in texts:
                    try:
                        # 生成文件名
                        title = text[:30].replace(' ', '_').replace('/', '_').replace('\n', ' ')
                        fname_new = f"{nid:06d}_{title[:50]}.txt"
                        fpath_new = os.path.join(KB_DIR, fname_new)
                        
                        with open(fpath_new, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        nid += 1
                        repo_count += 1
                        total_created += 1
                    except Exception as e:
                        total_failed += 1
                        continue
        
        if repo_count > 0:
            print(f"  {repo_dir}: +{repo_count}")
    
    print(f"\n✅ 总共创建: {total_created} 个知识文件")
    if total_failed > 0:
        print(f"❌ 失败: {total_failed}")
    print(f"📊 下一个ID: {nid}")

if __name__ == '__main__':
    main()
