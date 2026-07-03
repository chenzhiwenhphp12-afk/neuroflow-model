import json

path = r'D:\neuroflow-C++\configs\tokenizer_128k.json'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Simulate C++ parse_config
vocab_pos = content.find('"vocab": {')
if vocab_pos == -1:
    vocab_pos = content.find('"vocab":{')
print('vocab_pos:', vocab_pos)

# Find the opening brace of vocab object
obj_start = content.find('{', vocab_pos)
# Find matching closing brace
brace_depth = 0
in_string = False
escape_next = False
obj_end = obj_start
for obj_end in range(obj_start, len(content)):
    c = content[obj_end]
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
            break

print(f'obj_start: {obj_start}, obj_end: {obj_end}')
vocab_section = content[obj_start + 1:obj_end]
print(f'vocab_section length: {len(vocab_section)}')

# Now simulate the key-value parsing
parsed = {}
pos = 0
errors = 0
last_key = ''
while pos < len(vocab_section):
    key_start = vocab_section.find('"', pos)
    if key_start == -1:
        break
    key_end = vocab_section.find('"', key_start + 1)
    if key_end == -1:
        break
    
    key = vocab_section[key_start + 1:key_end - key_start - 1 + key_start + 1]
    
    colon = vocab_section.find(':', key_end)
    if colon == -1:
        break
    
    val_start = colon + 1
    while val_start < len(vocab_section) and vocab_section[val_start] in ' \t':
        val_start += 1
    
    val_end = val_start
    while val_end < len(vocab_section) and vocab_section[val_end] not in ',}':
        val_end += 1
    
    val_str = vocab_section[val_start:val_end].strip()
    
    try:
        token_id = int(val_str)
        parsed[key] = token_id
        last_key = key
    except:
        errors += 1
        if errors <= 5:
            print(f'Error parsing: key={repr(key)} val={repr(val_str)}')
    
    pos = val_end + 1

print(f'Parsed tokens: {len(parsed)}')
print(f'Errors: {errors}')
print(f'Last key: {repr(last_key)}')
if parsed:
    max_id = max(parsed.values())
    print(f'Max ID: {max_id}')
    print(f'Expected vocab_size: {max_id + 1}')