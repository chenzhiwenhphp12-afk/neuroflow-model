#!/usr/bin/env python3
"""云端运行: 从 prompts.json 调用 DeepSeek API 生成教师数据
用法: export ANTHROPIC_AUTH_TOKEN=sk-...
      python3 gen_from_prompts.py --prompts prompts.json --output teacher_data.jsonl
"""
import argparse, json, os, sys, time, requests

API_URL = os.environ.get('ANTHROPIC_BASE_URL',
    'https://api.deepseek.com/anthropic') + '/v1/messages'
API_KEY = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')

def call_api(prompt, max_tokens=128):
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
        return {'success': False, 'error': f'HTTP {resp.status_code}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', default='teacher_data.jsonl')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--resume', action='store_true', help='断点续传')
    args = parser.parse_args()

    if not API_KEY:
        print("❌ 请设置 ANTHROPIC_AUTH_TOKEN")
        sys.exit(1)

    # 加载prompts
    with open(args.prompts) as f:
        all_prompts = json.load(f)
    print(f"📝 {len(all_prompts)} prompts")

    # 断点续传
    completed = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    completed.add(r['prompt'])
                except: pass
        print(f"   已完成: {len(completed)}, 跳过")

    mode = 'a' if completed else 'w'
    success, total, skipped = 0, 0, 0
    t0 = time.time()

    with open(args.output, mode, encoding='utf-8') as out:
        for rec in all_prompts:
            prompt = rec['prompt']
            subj = rec.get('subject', '')

            if prompt in completed:
                skipped += 1
                continue

            result = call_api(prompt, args.max_tokens)
            total += 1

            if result['success'] and len(result['text']) >= 10:
                out.write(json.dumps({
                    'subject': subj, 'prompt': prompt,
                    'completion': result['text']
                }, ensure_ascii=False) + '\n')
                out.flush()
                success += 1

            if total % 20 == 0:
                elapsed = time.time() - t0
                rate = (success + skipped) * 60 / elapsed if elapsed > 0 else 0
                eta_h = (len(all_prompts) - success - skipped) / rate / 60 if rate > 0 else 0
                print(f"  [{success+skipped}/{len(all_prompts)}] "
                      f"ok={success} skip={skipped} rate={rate:.0f}/min eta={eta_h:.1f}h")

    elapsed = time.time() - t0
    size = os.path.getsize(args.output) / 1e6
    print(f"\n✅ 完成! {success}/{total} ok, {skipped} skipped")
    print(f"   {elapsed:.0f}s, {size:.1f}MB → {args.output}")


if __name__ == '__main__':
    main()
