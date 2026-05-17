"""从新仓库提取知识"""
import os, sys
sys.path.insert(0, '/mnt/d/neuroflow-model')
from extract_knowledge import process_file, get_file_id, KB_DIR

nid = get_file_id()
total = 0

for repo in ['ml-course-notes', 'deeplearningbook-chinese']:
    repo_path = f'/mnt/d/github_resources/{repo}'
    count = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d != '.git' and d != 'node_modules']
        for fname in files:
            if not fname.endswith(('.md', '.txt')):
                continue
            fpath = os.path.join(root, fname)
            texts = process_file(fpath)
            for text in texts:
                try:
                    title = text[:30].replace(' ', '_').replace('/', '_').replace('\n', ' ')
                    fname_new = f'{nid:06d}_{title[:50]}.txt'
                    with open(os.path.join(KB_DIR, fname_new), 'w', encoding='utf-8') as f:
                        f.write(text)
                    nid += 1
                    count += 1
                    total += 1
                except:
                    pass
    print(f'  {repo}: +{count}')

print(f'Total new: {total}')
print(f'Next ID: {nid}')
