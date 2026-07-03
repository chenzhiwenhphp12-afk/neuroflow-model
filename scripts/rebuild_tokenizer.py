import json

path = r'D:\neuroflow-C++\configs\tokenizer_128k.json'
d = json.load(open(path, 'r', encoding='utf-8'))
vocab = d['vocab']

# Rebuild vocab with sorted IDs (0, 1, 2, ..., 127999)
# This ensures JSON serialization puts them in order
sorted_vocab = {}
id_to_key = {}
for k, v in vocab.items():
    id_to_key[v] = k

for i in range(128000):
    if i in id_to_key:
        sorted_vocab[id_to_key[i]] = i
    else:
        sorted_vocab['<pad_extra_' + str(i) + '>'] = i

d['vocab'] = sorted_vocab
d['vocab_size'] = 128000

# Verify
assert len(sorted_vocab) == 128000
all_ids = set(sorted_vocab.values())
assert all_ids == set(range(128000))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print('Re-saved tokenizer_128k.json')
print('Total tokens:', len(sorted_vocab))
print('ID coverage:', min(sorted_vocab.values()), '-', max(sorted_vocab.values()))