#!/usr/bin/env python3
"""
NeuroFlow 全面蒸馏数据生成 — 从全学科语料提取 prompt，教师生成 completion

特点:
  - 扫描 corpus 所有目录，按学科分层采样
  - 从实际语料文件中提取文本作为 prompt
  - 每个 prompt 调用 DeepSeek API 生成教师回答
  - 覆盖: 小学(5科) + 初中(9科) + 高中(10科) + 大学(14学科+)

用法:
  export ANTHROPIC_AUTH_TOKEN=sk-...
  python3 scripts/gen_distill_full.py \
    --corpus ~/corpus --output ~/distill_data \
    --max-tokens 128 --timeout-min 240
"""

import argparse, json, os, random, sys, time, requests
from pathlib import Path

API_URL = os.environ.get('ANTHROPIC_BASE_URL',
                         'https://api.deepseek.com/anthropic') + '/v1/messages'
API_KEY = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')

# ═══ 领域映射 ═══
SUBJECT_MAP = {
    '语文': 'chinese', '外语': 'english', '数学': 'math', '科学': 'science',
    '物理': 'physics', '化学': 'chemistry', '生物': 'biology',
    '地理': 'geography', '历史': 'history', '政治': 'politics',
    '哲学': 'philosophy', '经济学': 'economics', '法学': 'law',
    '教育学': 'education', '文学': 'literature', '历史学': 'history',
    '理学': 'science', '工学': 'engineering', '农学': 'agriculture',
    '医学': 'medicine', '军事学': 'military', '管理学': 'management',
    '艺术学': 'arts', '交叉学科': 'interdisciplinary',
    '计算机': 'cs', '代码': 'code', '编程': 'programming',
}


def call_api(prompt: str, max_tokens: int = 128) -> dict:
    """返回 {text, success, error}"""
    try:
        resp = requests.post(API_URL, headers={
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
        }, json={
            'model': 'deepseek-chat',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0.7,
        }, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            for c in data.get('content', []):
                if c.get('type') == 'text':
                    return {'text': c['text'], 'success': True}
            if 'choices' in data:
                return {'text': data['choices'][0]['message']['content'], 'success': True}
            return {'success': False, 'error': f'No text in: {str(data)[:200]}'}
        return {'success': False, 'error': f'HTTP {resp.status_code}: {resp.text[:200]}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_prompts_from_file(filepath: Path, max_prompts: int = 10) -> list:
    """从文件中提取文本片段作为 prompt"""
    prompts = []
    try:
        ext = filepath.suffix.lower()
        if ext == '.jsonl':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        for k in ('text', 'content', 'title', 'question'):
                            if k in rec and isinstance(rec[k], str) and len(rec[k]) >= 10:
                                prompts.append(rec[k][:200])
                                break
                    except (json.JSONDecodeError, KeyError):
                        pass
                    if len(prompts) >= max_prompts:
                        break
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10 * 1024 * 1024)  # First 10MB only
            for pos in range(0, len(content) - 10, max(len(content) // (max_prompts * 2), 100)):
                snippet = content[pos:pos + 200].strip()
                if len(snippet) >= 20 and not snippet.startswith('{'):
                    prompts.append(snippet)
            prompts = random.sample(prompts, min(len(prompts), max_prompts)) if prompts else []
        elif ext in ('.txt', '.md'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024 * 1024)
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) >= 20]
            random.shuffle(paragraphs)
            prompts = paragraphs[:max_prompts]
        elif ext in ('.csv', '.tsv'):
            delim = '\t' if ext == '.tsv' else ','
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines(10000)
            for line in lines:
                fields = line.strip().split(delim)
                for fld in fields:
                    if len(fld) >= 20:
                        prompts.append(fld[:200])
            random.shuffle(prompts)
            prompts = prompts[:max_prompts]
    except Exception:
        pass
    return prompts


def scan_corpus(corpus_root: str) -> dict:
    """扫描语料库，按学科分组，返回 {subject: [file_paths]}"""
    root = Path(corpus_root)
    if not root.exists():
        print(f"❌ 语料目录不存在: {corpus_root}")
        sys.exit(1)

    subjects = {}
    for filepath in root.rglob('*'):
        if not filepath.is_file():
            continue
        ext = filepath.suffix.lower()
        if ext not in ('.txt', '.json', '.jsonl', '.csv', '.tsv', '.md'):
            continue

        # 推断学科: 看路径中的目录名
        subj = '通用'
        for part in filepath.parts:
            for key in SUBJECT_MAP:
                if key in part:
                    subj = key
                    break
        # 也看文件名
        fname = filepath.stem
        for key in SUBJECT_MAP:
            if key in fname:
                subj = key
                break

        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(filepath)

    return subjects


def main():
    parser = argparse.ArgumentParser(description='NeuroFlow 全面蒸馏数据生成')
    parser.add_argument('--corpus', default='/home/administrator/corpus', help='语料根目录')
    parser.add_argument('--output', default='/home/administrator/distill_data', help='输出目录')
    parser.add_argument('--max-tokens', type=int, default=128, help='教师生成长度')
    parser.add_argument('--samples-per-subject', type=int, default=0,
                        help='每学科样本上限(0=自动)')
    parser.add_argument('--timeout-min', type=int, default=480, help='总超时(分钟)')
    args = parser.parse_args()

    if not API_KEY:
        print("❌ 请设置环境变量 ANTHROPIC_AUTH_TOKEN")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 0. 断点续传: 加载已完成的 prompt
    out_file = os.path.join(args.output, 'teacher_data.jsonl')
    completed = set()
    if os.path.exists(out_file):
        print("📂 加载已完成数据...")
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    completed.add((rec.get('subject', ''), rec['prompt']))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"   已完成: {len(completed)} 样本，断点续传")

    # 1. 扫描语料
    print("🔍 扫描语料库...")
    subjects = scan_corpus(args.corpus)
    total_files = sum(len(v) for v in subjects.values())
    print(f"   {len(subjects)} 个学科, {total_files:,} 个文件")
    for name, files in sorted(subjects.items(), key=lambda x: -len(x[1])):
        print(f"     {name}: {len(files)} 文件")

    # 2. 每学科采样 prompt (固定种子确保断点可复现)
    print("\n📝 提取 prompts...")
    random.seed(42)  # 固定种子，确保断点重入提取相同prompts
    all_prompts = {}
    total_expected = 0
    for subj, files in sorted(subjects.items()):
        limit = args.samples_per_subject or max(50, len(files) // 10)
        prompts = []
        random.shuffle(files)
        for fp in files[:min(len(files), max(100, limit))]:
            extracted = extract_prompts_from_file(fp, max_prompts=5)
            prompts.extend(extracted)
            if len(prompts) >= limit:
                break
        all_prompts[subj] = prompts[:limit]
        total_expected += len(all_prompts[subj])
        print(f"   {subj}: {len(all_prompts[subj])} prompts (目标{limit})")
    print(f"   总计: {total_expected} prompts")

    # 3. 调用 API 生成教师数据
    print(f"\n🤖 调用 DeepSeek API ({len(completed)} 已跳过)...")
    t0 = time.time()
    deadline = t0 + args.timeout_min * 60
    out_file = os.path.join(args.output, 'teacher_data.jsonl')
    stats_file = os.path.join(args.output, 'stats.json')

    total = 0
    success = 0
    total_chars = 0

    skipped = 0
    with open(out_file, 'a', encoding='utf-8') as fout:  # append mode
        for subj, prompts in sorted(all_prompts.items()):
            for i, prompt in enumerate(prompts):
                # 断点续传: 跳过已完成
                if (subj, prompt) in completed:
                    skipped += 1
                    continue
                if time.time() > deadline:
                    print(f"\n⏰ 超时 ({args.timeout_min}min)，已生成 {total} 样本")
                    break

                result = call_api(prompt, args.max_tokens)
                total += 1

                if result['success'] and len(result['text']) >= 20:
                    rec = {
                        'subject': subj,
                        'prompt': prompt[:300],
                        'completion': result['text'],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    fout.flush()
                    success += 1
                    total_chars += len(result['text'])
                else:
                    err = result.get('error', 'unknown')[:100]
                    print(f"   ❌ {subj}[{i}]: {err}")

                if total % 20 == 0:
                    elapsed = time.time() - t0
                    rate = total * 60 / elapsed if elapsed > 0 else 0
                    print(f"   [{total}] {subj}: {success} ok, {rate:.0f}/min, "
                          f"{elapsed:.0f}s elapsed")

                # Rate limit
                if total % 80 == 0:
                    time.sleep(0.5)

            if time.time() > deadline:
                break

    elapsed = time.time() - t0

    # 4. 统计
    print(f"\n{'='*60}")
    print(f"✅ 蒸馏数据生成完成!")
    print(f"{'='*60}")
    print(f"   样本数: {success}/{total}")
    print(f"   字符数: {total_chars:,}")
    print(f"   耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   速率: {success*60/elapsed:.0f} samples/min")

    fees = success * args.max_tokens / 1e6 * 0.28 + total * 200 / 1e6 * 0.14
    print(f"   费用: ~${fees:.2f}")
    print(f"   输出: {out_file}")

    with open(stats_file, 'w') as f:
        json.dump({
            'samples': success, 'total_calls': total,
            'chars': total_chars, 'elapsed': elapsed,
            'subjects': {k: len(v) for k, v in all_prompts.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"   统计: {stats_file}")

    if success == 0:
        print("\n⚠️ 没有成功生成任何样本! 检查 API Key 和网络")


if __name__ == '__main__':
    main()
