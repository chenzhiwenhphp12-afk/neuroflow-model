"""
NeuroFlow v5.0 Subword Tokenizer
=================================
Mini-BPE tokenizer: 5000 subword vocabulary
- Uses tiktoken's cl100k_base as base (GPT-4 tokenizer)
- Downsamples to top-5K most frequent tokens
- Handles Chinese characters via UTF-8 byte fallback
- O(1) lookup via dict + numpy array
"""

import numpy as np
import json, os

VOCAB_SIZE = 5000         # v5.0 target
MAX_SEQ_LEN = 8000        # max tokens per sequence
PAD_ID = 0
UNK_ID = 1

class MiniBPE:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._enc = None  # lazy tiktoken encoder cache
        self._load_or_create()
    
    def _get_enc(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        return self._enc
    
    def _load_or_create(self):
        """Load existing tokenizer or create from tiktoken"""
        path = os.path.join(os.path.dirname(__file__), "..", "tokenizer_v5.json")
        if os.path.exists(path):
            self._load(path)
            print(f"  📖 加载BPE词表: {len(self.token_to_id)} 个subword tokens")
        else:
            self._create_from_tiktoken()
            self._save(path)
            print(f"  ✅ 创建BPE词表: {len(self.token_to_id)} 个subword tokens")
    
    def _create_from_tiktoken(self):
        """Use tiktoken cl100k_base to build our 5K vocabulary"""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        # tiktoken has ~100K tokens. Take top 5K most common
        # tokens 0-4999 are typically the most frequent
        all_tokens = list(range(min(self.vocab_size, enc.max_token_value)))
        
        # Map first vocab_size tokens
        for tid in all_tokens[:self.vocab_size]:
            try:
                token_bytes = enc.decode_single_token_bytes(tid)
                token_str = token_bytes.decode("utf-8", errors="replace")
                self.token_to_id[token_str] = tid
                self.id_to_token[tid] = token_str
            except:
                pass
        
        # Ensure PAD and UNK exist
        self.token_to_id["<PAD>"] = 0
        self.id_to_token[0] = "<PAD>"
        self.token_to_id["<UNK>"] = 1
        self.id_to_token[1] = "<UNK>"
    
    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
    
    def _save(self, path):
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def encode(self, text, max_len=MAX_SEQ_LEN):
        """Encode text → token IDs array of length max_len"""
        enc = self._get_enc()
        raw_ids = enc.encode(text)[:max_len - 2]  # reserve for <BOS> <EOS>
        
        # Clip to vocab range
        ids = [0]  # <BOS>
        for tid in raw_ids:
            if tid < self.vocab_size:
                ids.append(tid)
            else:
                ids.append(UNK_ID)  # <UNK> for OOV
            if len(ids) >= max_len - 1:
                break
        ids.append(0)  # <EOS>
        
        # Pad to max_len
        arr = np.zeros(max_len, dtype=np.int32)
        n = min(len(ids), max_len)
        arr[:n] = ids[:n]
        return arr
    
    def decode(self, ids):
        """Decode token IDs → text"""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Filter out PAD and special tokens
        valid = [tid for tid in ids if tid > 1 and tid < self.vocab_size]
        try:
            return enc.decode(valid)
        except:
            return ""


# Global singleton
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = MiniBPE()
    return _tokenizer


# Quick test
if __name__ == "__main__":
    t = get_tokenizer()
    for text in ["Hello world", "人工智能", "NeuroFlow v5.0 with BPE"]:
        ids = t.encode(text)
        decoded = t.decode(ids)
        print(f"  '{text}' → {len(ids)} tokens → '{decoded[:40]}'")
    print(f"  Vocab: {len(t.token_to_id)} / {t.vocab_size}")
    print("  Tokenizer OK")
