#!/usr/bin/env python3
"""
NeuroFlow 蒸馏数据生成 — 调用 DeepSeek API 获取教师数据

用法:
  python3 scripts/gen_distill_data.py --prompts prompts.txt --output teacher_data.jsonl --max-samples 10000

输出 JSONL 格式: {"prompt": "...", "completion": "...", "teacher_text": "..."}
"""

import argparse, json, os, sys, time, random
import requests

# DeepSeek API (Anthropic-compatible endpoint)
API_URL = os.environ.get('DEEPSEEK_API_URL',
    'https://api.deepseek.com/anthropic/v1/messages')
API_KEY = os.environ.get('DEEPSEEK_API_KEY',
    os.environ.get('ANTHROPIC_AUTH_TOKEN', ''))
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def call_deepseek(prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
    """Call DeepSeek API to generate text completion"""
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            # Extract text from response
            for choice in data.get("content", []):
                if choice.get("type") == "text":
                    return choice.get("text", "")
            # Fallback: try standard format
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        else:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return ""
    except Exception as e:
        print(f"  API exception: {e}")
        return ""


def load_prompts(path: str) -> list:
    """Load prompts from file (one per line) or generate default Chinese prompts"""
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Default: diverse Chinese prompts from different domains
    defaults = [
        # 科学
        "请解释量子力学的基本原理",
        "什么是相对论？用简单的话解释",
        "人工智能的发展历史是什么",
        "DNA双螺旋结构的发现过程",
        # 数学
        "如何求解一元二次方程",
        "微积分的基本思想是什么",
        "请解释概率论中的贝叶斯定理",
        # 文学
        "请写一首关于春天的诗",
        "唐诗和宋词的区别是什么",
        "中国四大名著有哪些",
        # 历史
        "秦始皇统一六国的过程",
        "工业革命对世界的影响",
        # 生活
        "如何保持健康的饮食习惯",
        "怎样提高学习效率",
        "推荐几种有效的记忆方法",
        # 哲学
        "什么是存在主义",
        "孔子的仁政思想是什么",
        # 技术
        "Python编程语言的特点",
        "什么是深度学习",
        "区块链技术的基本原理",
    ]
    return defaults


def main():
    parser = argparse.ArgumentParser(description='NeuroFlow 蒸馏数据生成')
    parser.add_argument('--prompts', default='', help='Prompt文件 (一行一个)')
    parser.add_argument('--output', default='teacher_data.jsonl', help='输出文件')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数')
    parser.add_argument('--max-tokens', type=int, default=128, help='教师生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='教师温度')
    parser.add_argument('--repeat', type=int, default=1, help='每个prompt重复次数(不同温度)')
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"📝 加载 {len(prompts)} 个 prompts")

    # Expand: each prompt × repeat times with different temperatures
    all_tasks = []
    for p in prompts:
        for r in range(args.repeat):
            temp = 0.3 + r * 0.4 / max(1, args.repeat - 1) if args.repeat > 1 else args.temperature
            all_tasks.append((p, temp))

    random.shuffle(all_tasks)
    all_tasks = all_tasks[:args.max_samples]

    print(f"🎯 目标: {len(all_tasks)} 个样本")
    print(f"🔑 API: {API_URL}")
    print()

    success = 0
    total_chars = 0
    t0 = time.time()

    with open(args.output, 'w', encoding='utf-8') as out:
        for i, (prompt, temp) in enumerate(all_tasks):
            text = call_deepseek(prompt, args.max_tokens, temp)

            if text and len(text) >= 10:
                record = {
                    "prompt": prompt,
                    "completion": text,
                    "temperature": temp,
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                success += 1
                total_chars += len(text)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    print(f"  [{i+1}/{len(all_tasks)}] success={success} "
                          f"rate={rate:.0f}/min elapsed={elapsed:.0f}s")
            else:
                print(f"  [{i+1}] ❌ empty response for: {prompt[:50]}")

            # Rate limit: ~60 requests/min to be safe
            if (i + 1) % 50 == 0:
                time.sleep(1)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6 if os.path.exists(args.output) else 0

    print(f"\n✅ 完成!")
    print(f"   样本: {success}/{len(all_tasks)}")
    print(f"   字符: {total_chars:,}")
    print(f"   文件: {args.output} ({size_mb:.1f} MB)")
    print(f"   耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   费用预估: ~${success * 128 / 1e6 * 0.14:.2f} (DeepSeek 输入$0.14/M tokens)")

    # Save stats
    stats = {
        "total_samples": success,
        "total_chars": total_chars,
        "elapsed_seconds": elapsed,
        "model": "deepseek-chat",
    }
    stats_path = args.output.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   统计: {stats_path}")


if __name__ == '__main__':
    main()
