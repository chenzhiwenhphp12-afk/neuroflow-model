import json

path = r'D:\neuroflow-C++\configs\tokenizer_128k.json'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find vocab section
vocab_pos = content.find('"vocab": {')
obj_start = content.find('{', vocab_pos)

# Simulate C++ parsing more closely
pos = obj_start + 1
obj_limit = len(content)  # simplified
parsed = {}
errors = 0
last_good_pos = 0

while pos < obj_limit:
    key_start = content.find('"', pos)
    if key_start == -1 or key_start >= obj_limit:
        break
    key_end = content.find('"', key_start + 1)
    if key_end == -1 or key_end >= obj_limit:
        break
    
    key = content[key_start + 1:key_end]
    
    # Check if key contains special chars that might confuse parser
    if '\\' in key or '"' in key[1:]:
        print(f'Suspicious key at pos {key_start}: {repr(key[:50])}')
    
    colon = content.find(':', key_end)
    if colon == -1 or colon >= obj_limit:
        break
    
    val_start = colon + 1
    while val_start < obj_limit and content[val_start] in ' \t\n\r':
        val_start += 1
    
    val_end = val_start
    while val_end < obj_limit and content[val_end] not in ',}\n\r':
        val_end += 1
    
    val_str = content[val_start:val_end].strip()
    
    try:
        token_id = int(val_str)
        parsed[key] = token_id
    except:
        errors += 1
        if errors <= 5:
            print(f'Parse error: key={repr(key[:30])} val={repr(val_str[:30])}')
    
    last_good_pos = val_end
    pos = val_end + 1

print(f'Total parsed: {len(parsed)}')
print(f'Errors: {errors}')
print(f'Last good position: {last_good_pos}')

# Check what's at last_good_pos
if last_good_pos < len(content):
    print(f'Content around last good pos: {repr(content[last_good_pos:last_good_pos+100])}')

# Find the actual closing brace of vocab
brace_depth = 0
in_string = False
escape_next = False
for i in range(obj_start, len(content)):
    c = content[i]
    if escape_next:
        escape_next = False
        continue
    if c == '\\':
        escape_next = True
        continue
    if c == '"':
        in_string = not in_string
        continue
    if in_string:
        continue
    if c == '{':
        brace_depth += 1
    elif c == '}':
        brace_depth -= 1
        if brace_depth == 0:
            print(f'Actual vocab object end: {i}')
            print(f'Content around end: {repr(content[max(0,i-50):i+10])}')
            break