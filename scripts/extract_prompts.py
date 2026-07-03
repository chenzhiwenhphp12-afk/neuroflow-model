#!/usr/bin/env python3
"""本地扫描提取所有 prompts，存为便携文件 → 上传云端"""
import json, os, random, sys
from pathlib import Path

SUBJECT_MAP = {
    '语文':'chinese','外语':'english','数学':'math','科学':'science',
    '物理':'physics','化学':'chemistry','生物':'biology',
    '地理':'geography','历史':'history','政治':'politics',
    '哲学':'philosophy','经济学':'economics','法学':'law',
    '教育学':'education','文学':'literature','历史学':'history',
    '理学':'science','工学':'engineering','农学':'agriculture',
    '医学':'medicine','军事学':'military','管理学':'management',
    '艺术学':'arts','交叉学科':'interdisciplinary',
}

random.seed(42)

def extract_prompts(filepath, max_prompts=10):
    prompts = []
    try:
        ext = filepath.suffix.lower()
        if ext == '.jsonl':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        for k in ('text','content','title','question'):
                            if k in r and isinstance(r[k],str) and len(r[k])>=10:
                                prompts.append(r[k][:200])
                                break
                    except: pass
                    if len(prompts)>=max_prompts: break
        elif ext == '.json':
            with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
                content = f.read(10*1024*1024)
            snippets = [content[i:i+200].strip() for i in range(0,min(len(content),100000),500)]
            prompts = [s for s in snippets if len(s)>=20 and not s.startswith('{')]
            random.shuffle(prompts)
            prompts = prompts[:max_prompts]
        elif ext in ('.txt','.md'):
            with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
                content = f.read(1024*1024)
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip())>=20]
            random.shuffle(paragraphs)
            prompts = paragraphs[:max_prompts]
        elif ext in ('.csv','.tsv'):
            delim = '\t' if ext=='.tsv' else ','
            with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
                lines = f.readlines(10000)
            for line in lines:
                for fld in line.strip().split(delim):
                    if len(fld)>=20: prompts.append(fld[:200])
            prompts = prompts[:max_prompts]
    except: pass
    return prompts

def scan_corpus(root):
    root = Path(root)
    subjects = {}
    for fp in root.rglob('*'):
        if not fp.is_file(): continue
        if fp.suffix.lower() not in ('.txt','.json','.jsonl','.csv','.tsv','.md'): continue
        subj = '通用'
        for part in fp.parts:
            for k in SUBJECT_MAP:
                if k in part: subj = k; break
        fname = fp.stem
        for k in SUBJECT_MAP:
            if k in fname: subj = k; break
        subjects.setdefault(subj, []).append(fp)
    return subjects

def main():
    corpus = sys.argv[1] if len(sys.argv)>1 else os.path.expanduser('~/corpus')
    output = sys.argv[2] if len(sys.argv)>2 else os.path.expanduser('~/prompts.json')

    print(f'🔍 扫描: {corpus}')
    subjects = scan_corpus(corpus)
    total = sum(len(v) for v in subjects.values())
    print(f'   {len(subjects)} 学科, {total:,} 文件')

    all_prompts = []
    for subj, files in sorted(subjects.items()):
        limit = len(files) // 10
        prompts = []
        random.shuffle(files)
        for fp in files[:min(len(files), max(100, limit))]:
            prompts.extend(extract_prompts(fp, 5))
            if len(prompts) >= limit: break
        for p in prompts[:limit]:
            all_prompts.append({'subject': subj, 'prompt': p})
        print(f'   {subj}: {len(prompts[:limit])} prompts')

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, ensure_ascii=False)
    size_mb = os.path.getsize(output)/1e6
    print(f'\n✅ {len(all_prompts)} prompts → {output} ({size_mb:.1f}MB)')
    print(f'   上传此文件到云服务器，运行 gen_distill_from_prompts.py')

if __name__ == '__main__':
    main()
