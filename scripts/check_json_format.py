import json

path = r'D:\neuroflow-C++\configs\tokenizer_128k.json'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

print('Lines:', content.count('\n'))
print('Size:', len(content))

# Find vocab section
idx = content.find('"vocab"')
print('vocab key at position:', idx)
if idx > 0:
    print('Context:', repr(content[idx:idx+50]))

# The C++ parse_config looks for "\"vocab\":{" or "\"vocab\": {"
# With indent=2, it's likely "\"vocab\": {"
pattern1 = '"vocab": {'
pattern2 = '"vocab":{'
p1 = content.find(pattern1)
p2 = content.find(pattern2)
print(f'Pattern "{pattern1}" found at:', p1)
print(f'Pattern "{pattern2}" found at:', p2)