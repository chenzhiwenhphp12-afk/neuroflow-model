import json

d = json.load(open(r'D:\neuroflow-C++\configs\tokenizer_128k.json', 'r', encoding='utf-8'))
vocab = d['vocab']

# Check padded tokens
missing_range = list(range(127289, 127526))
found = 0
for mid in missing_range:
    for k, v in vocab.items():
        if v == mid:
            found += 1
            break
print('Found', found, 'out of', len(missing_range), 'tokens in missing range 127289-127525')

# Check unused and extra tokens
unused = {k:v for k,v in vocab.items() if k.startswith('<unused_')}
extra = {k:v for k,v in vocab.items() if k.startswith('<extra_')}
print('unused_ tokens:', len(unused))
print('extra_ tokens:', len(extra))

# Check specific tokens
for tid in [127289, 127525, 127526, 127999]:
    for k, v in vocab.items():
        if v == tid:
            print('ID', tid, '->', repr(k))
            break
    else:
        print('ID', tid, '-> NOT FOUND')

# Check: does the raw JSON file have these tokens?
with open(r'D:\neuroflow-C++\configs\tokenizer_128k.json', 'r', encoding='utf-8') as f:
    raw = f.read()

# Find the vocab closing brace
vocab_close = raw.find('"vocab": {')
# Find the closing }
brace_depth = 0
in_str = False
esc = False
end_pos = 0
for i in range(vocab_close, len(raw)):
    c = raw[i]
    if esc: esc = False; continue
    if c == '\\': esc = True; continue
    if c == '"': in_str = not in_str; continue
    if in_str: continue
    if c == '{': brace_depth += 1
    elif c == '}':
        brace_depth -= 1
        if brace_depth == 0:
            end_pos = i
            break

print('Vocab object ends at position:', end_pos)
print('Content around end:', repr(raw[end_pos-80:end_pos+20]))

# Check: are there tokens AFTER the vocab closing brace?
after = raw[end_pos+1:end_pos+200]
print('After vocab close:', repr(after[:100]))