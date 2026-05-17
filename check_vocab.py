import json

vocab = json.load(open('/mnt/d/neuroflow-model/char_vocab.json'))
print(f'当前词表: {len(vocab)} 个字符')
print('Top 30:', repr(''.join(vocab[:30])))

# 统计类型
en = sum(1 for c in vocab if c.isascii() and c.isalpha())
digits = sum(1 for c in vocab if c.isdigit())
punct = sum(1 for c in vocab if c in '.,;:!?()[]{}-\'"<>/|_#+=*&^%$@~`')
cn = sum(1 for c in vocab if ord(c) > 0x4E00)
other = len(vocab) - en - digits - punct - cn
print(f'类型: 英文={en} 数字={digits} 标点={punct} 中文={cn} 其他={other}')

for size in [300, 500, 800, 1000]:
    v = vocab[:size]
    en_c = sum(1 for c in v if c.isascii() and c.isalpha())
    cn_c = sum(1 for c in v if ord(c) > 0x4E00)
    print(f'前{size}: 英文={en_c} 中文={cn_c} 其他={size-en_c-cn_c}')
