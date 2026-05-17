"""重建 char_vocab.json 为前500高频字符"""
import json

vocab = json.load(open('/mnt/d/neuroflow-model/char_vocab.json'))
new_vocab = vocab[:500]

# 确保基本的英文字母+数字+常见标点都在
required = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}/")
for ch in required:
    if ch not in new_vocab:
        new_vocab.append(ch)
        print(f"  + 添加缺失: {repr(ch)}")

# 去重（确保唯一）
seen = set()
deduped = []
for ch in new_vocab:
    if ch not in seen:
        seen.add(ch)
        deduped.append(ch)

# 如果去重后少于500，再加入高频中文
if len(deduped) < 500:
    for ch in vocab:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)
        if len(deduped) >= 500:
            break

new_vocab = deduped[:500]
json.dump(new_vocab, open('/mnt/d/neuroflow-model/char_vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'✅ 新词表: {len(new_vocab)} 字符')

# 验证类型
en = sum(1 for c in new_vocab if c.isascii() and c.isalpha())
digits = sum(1 for c in new_vocab if c.isdigit())
cn = sum(1 for c in new_vocab if ord(c) > 0x4E00)
print(f'  英文={en} 数字={digits} 中文={cn} 其他={500-en-digits-cn}')
print(f'  前30: {"".join(new_vocab[:30])}')
