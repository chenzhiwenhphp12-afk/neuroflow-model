import json

path = r'D:\neuroflow-C++\configs\tokenizer_128k.json'
d = json.load(open(path, 'r', encoding='utf-8'))
vocab = d['vocab']

expected = set(range(128000))
actual = set(vocab.values())
missing = sorted(list(expected - actual))

print('Missing IDs:', len(missing))
print('Range:', missing[0], '-', missing[-1])

for mid in missing:
    token = '<unused_' + str(mid) + '>'
    if token in vocab:
        print('Collision at', token, '-> using alt')
        token = '<fill_' + str(mid) + '>'
    vocab[token] = mid

d['vocab'] = vocab
d['vocab_size'] = 128000

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

# Verify
all_ids = set(vocab.values())
still_missing = set(range(128000)) - all_ids
print('After fix - tokens:', len(vocab), 'missing IDs:', len(still_missing))
