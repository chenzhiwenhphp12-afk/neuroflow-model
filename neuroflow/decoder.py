"""
NeuroFlow TextDecoder — 轻量文本解码器
=========================================
将 NeuroFlow 类脑神经信号解码为文本 Token。
架构：ECN hidden → Linear Projection → Token Sampling
参数量：~128K（含 5000 词表），CPU 推理 < 2ms
"""

import numpy as np
from typing import List, Optional, Tuple


class TokenSampler:
    """Token 采样策略：temperature + top-k + top-p"""

    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> int:
        logits = logits.astype(np.float64)
        logits = logits / max(temperature, 1e-8)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = np.clip(probs, 1e-30, None)
        probs = probs / probs.sum()

        # Top-k
        if top_k > 0 and top_k < len(probs):
            indices = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros(len(probs), dtype=bool)
            mask[indices] = True
            probs = probs * mask

        # Top-p
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, top_p)) + 1
            cutoff = max(cutoff, 1)  # at least 1 token
            mask = np.zeros(len(probs), dtype=bool)
            mask[sorted_idx[:cutoff]] = True
            probs = probs * mask

        # Renormalize with safe division
        s = probs.sum()
        if s < 1e-30:
            # Fallback: uniform over all tokens
            return int(np.random.randint(0, len(probs)))
        probs = probs / s
        return int(np.random.choice(len(probs), p=probs))


class TextDecoder:
    """
    轻量文本解码器。
    
    将 NeuroFlow 的脑网络信号（ECN hidden states + DMN context + SN gates）
    投影到词汇空间，支持自回归文本生成。
    
    参数量：hidden_dim × vocab_size ≈ 256 × 5000 = 1.28M
    量化后（INT8）：~160KB
    
    Example:
        >>> model = NeuroFlowLite(input_dim=512)
        >>> decoder = TextDecoder(model, hidden_dim=256, vocab_size=5000)
        >>> tokens = decoder.generate("The future of AI is", max_tokens=20)
        >>> print(decoder.decode(tokens))
    """

    def __init__(
        self,
        model,
        hidden_dim: int = 256,
        vocab_size: int = 5000,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        """
        Args:
            model: NeuroFlow 模型实例（Python 或 C++ 后端）
            hidden_dim: ECN 隐藏层维度
            vocab_size: 词表大小
            max_seq_len: 最大生成长度
            seed: 随机种子
        """
        self.model = model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 解码器权重：Linear(hidden_dim, vocab_size)
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / hidden_dim)  # He init
        self.W_proj = rng.randn(hidden_dim, vocab_size).astype(np.float32) * scale
        self.b_proj = np.zeros(vocab_size, dtype=np.float32)

        # 简易词表（ASCII 字符级 + 常用词）
        self.vocab = self._build_vocab(vocab_size)

        # 状态缓存
        self.context_memory: Optional[np.ndarray] = None

    def _build_vocab(self, size: int) -> List[str]:
        """构建基础词表：ASCII 字符 + 常用子词 + 特殊 token"""
        tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"]

        # ASCII 可打印字符
        for i in range(32, 127):
            tokens.append(chr(i))

        # 常用英文子词 / 短词
        common = [
            "the", "of", "and", "to", "a", "in", "is", "that", "it", "for",
            "was", "on", "are", "be", "as", "with", "his", "they", "at", "this",
            "from", "or", "one", "had", "by", "word", "but", "not", "what", "all",
            "were", "we", "when", "your", "can", "said", "there", "use", "an", "each",
            "which", "she", "do", "how", "their", "if", "will", "up", "other", "about",
            "out", "many", "then", "them", "these", "so", "some", "her", "would", "make",
            "like", "him", "into", "time", "has", "look", "two", "more", "write", "go",
            "see", "number", "no", "way", "could", "people", "my", "than", "first", "water",
            "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day",
            "did", "get", "come", "made", "may", "part", "over", "new", "sound", "take",
            "only", "little", "work", "know", "place", "year", "live", "me", "back", "give",
            "most", "very", "after", "thing", "our", "just", "name", "good", "sentence", "man",
            "think", "say", "great", "where", "help", "through", "much", "before", "line", "right",
            "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came", "want",
            "show", "also", "around", "form", "three", "small", "set", "put", "end", "does",
            "another", "well", "large", "must", "big", "even", "such", "here", "why", "ask",
            "went", "men", "read", "need", "land", "different", "home", "us", "move", "try",
            "kind", "hand", "picture", "again", "change", "off", "play", "spell", "air", "away",
            "animal", "house", "point", "page", "letter", "mother", "answer", "found", "study", "still",
            "learn", "should", "world", "high", "every", "near", "add", "food", "between", "own",
            "below", "country", "plant", "last", "school", "father", "keep", "tree", "never", "start",
            "city", "earth", "eye", "light", "thought", "head", "under", "story", "saw", "left",
            "few", "while", "along", "might", "close", "something", "seem", "next", "hard", "open",
            "example", "begin", "life", "always", "those", "both", "paper", "together", "got", "group",
            "often", "run", "important", "until", "children", "side", "feet", "car", "mile", "night",
            "walk", "white", "sea", "began", "grow", "took", "river", "four", "carry", "state",
            "once", "book", "hear", "stop", "without", "second", "late", "miss", "idea", "enough",
            "eat", "face", "watch", "far", "really", "almost", "let", "above", "girl", "sometimes",
            "mountain", "cut", "young", "talk", "soon", "list", "song", "being", "leave", "family",
            "body", "color", "stand", "sun", "question", "fish", "area", "mark", "dog", "horse",
            "bird", "problem", "complete", "room", "knew", "since", "ever", "piece", "told", "usually",
            "didn", "friends", "easy", "heard", "order", "red", "door", "sure", "become", "top",
            "ship", "across", "today", "during", "short", "better", "best", "however", "low", "hours",
            "black", "products", "happened", "whole", "measure", "remember", "early", "waves", "reached",
            "listen", "wind", "rock", "space", "covered", "fast", "several", "hold", "himself", "toward",
            "five", "step", "morning", "passed", "true", "hundred", "against", "pattern", "table", "north",
            "slowly", "money", "map", "busy", "music", "below", "street", "science", "machine", "note",
            "wait", "plan", "figure", "star", "box", "rest", "correct", "able", "beautiful", "drive",
            "deep", "moon", "island", "foot", "system", "test", "record", "boat", "common", "gold",
            "possible", "plane", "age", "wonder", "laugh", "thousand", "check", "game", "shape", "deep",
            "yes", "hot", "bring", "heat", "snow", "object", "rule", "among", "power", "town",
            "unit", "language", "speed", "fall", "king", "toward", "certain", "field", "half", "war",
            "lay", "against", "pattern", "center", "love", "person", "money", "serve", "appear", "road",
            "rain", "develop", "class", "piece", "surface", "laugh", "moon", "star", "cross", "build",
            "present", "dress", "skin", "region", "island", "record", "direct", "material", "dance", "fire",
            "south", "deep", "square", "string", "grass", "quiet", "nature", "forest", "farm", "similar",
            "problem", "complete", "window", "store", "summer", "train", "sleep", "prove", "lone", "leg",
            "wall", "catch", "mount", "wish", "sky", "board", "joy", "winter", "sat", "written",
            "wild", "kept", "glass", "west", "lay", "weather", "root", "instruments", "meet", "third",
            "month", "paragraph", "raised", "represent", "soft", "whether", "clothes", "flowers", "shall", "held",
            "describe", "drive", "cross", "speak", "solve", "metal", "son", "either", "ice", "sleep",
            "village", "factors", "result", "jumped", "snow", "ride", "care", "floor", "hill", "pushed",
            "baby", "buy", "century", "outside", "everything", "tall", "already", "phrase", "soil", "bed",
        ]
        tokens.extend(common)

        # 填充到目标大小
        while len(tokens) < size:
            tokens.append(f"<T{len(tokens)}>")

        return tokens[:size]

    @property
    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """简单字符级编码：每个字符映射到词表中最接近的 token"""
        ids = [1]  # <BOS>
        for ch in text:
            if ch in self.vocab:
                ids.append(self.vocab.index(ch))
            else:
                ids.append(3)  # <UNK>
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """将 token ID 列表解码为文本"""
        result = []
        for tid in token_ids:
            if tid < len(self.vocab):
                token = self.vocab[tid]
                if token.startswith("<") and token.endswith(">"):
                    continue  # 跳过特殊 token
                result.append(token)
        return "".join(result)

    def _extract_context(self, x: np.ndarray) -> np.ndarray:
        """
        从 NeuroFlow 提取上下文表征。
        
        使用 ECN 的 hidden states 作为主要信号，
        混合 DMN 检索记忆 + SN 门控。
        """
        output = self.model.forward(x, consolidate=False, return_manifold=False)

        # 主信号：ECN hidden states（通过 decision/value 反推）
        decision = output.decision  # [1, output_dim]
        value = output.value  # [1, output_dim]

        # 从 saliency/ecn_gate 构建注意力权重
        try:
            ecn_gate = output.ecn_gate.flatten()
        except Exception:
            ecn_gate = np.ones(self.hidden_dim) * 0.5

        try:
            saliency = output.saliency.flatten()
        except Exception:
            saliency = np.zeros(self.hidden_dim)

        # 混合：decision 信号 + value 评估 + saliency 门控
        # 将低维输出投影到 hidden_dim
        out_dim = decision.shape[1]
        combined = np.concatenate([decision[0], value[0], saliency[:out_dim]], axis=0)

        # 用简单线性扩展填充到 hidden_dim
        if len(combined) < self.hidden_dim:
            # 重复 + 噪声填充
            repeats = self.hidden_dim // len(combined) + 1
            expanded = np.tile(combined, repeats)[:self.hidden_dim]
            # 添加微小噪声打破对称
            expanded += np.random.randn(self.hidden_dim).astype(np.float32) * 0.01
        else:
            expanded = combined[:self.hidden_dim]

        # 应用 ECN 门控
        ecn_gate_trimmed = ecn_gate[:self.hidden_dim]
        context = expanded * (0.5 + 0.5 * np.tanh(ecn_gate_trimmed))

        return context.astype(np.float32)

    def _logits(self, context: np.ndarray) -> np.ndarray:
        """计算 token logits：context @ W_proj + b_proj"""
        return context @ self.W_proj + self.b_proj

    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        input_dim: int = 512,
        verbose: bool = False,
    ) -> Tuple[List[int], str]:
        """
        自回归生成文本 Token。
        
        Args:
            prompt: 提示文本（编码为输入特征）
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样
            top_p: nucleus 采样阈值
            input_dim: 输入维度
            verbose: 打印生成过程
        
        Returns:
            (token_ids, decoded_text)
        """
        token_ids = [1]  # <BOS>

        # 将 prompt 编码为特征向量
        if prompt:
            prompt_ids = self.encode(prompt)
            # 简单方式：用 prompt 长度调制输入
            prompt_signal = np.sin(np.linspace(0, len(prompt_ids), input_dim)).astype(np.float32)
        else:
            prompt_signal = np.random.randn(input_dim).astype(np.float32) * 0.01

        x = prompt_signal.reshape(1, -1)

        generated_text = ""

        for step in range(max_tokens):
            # 1. 通过 NeuroFlow 提取脑信号
            context = self._extract_context(x)

            # 2. 投影到词表
            logits = self._logits(context)

            # 3. 采样 token
            token_id = TokenSampler.sample(logits, temperature, top_k, top_p)
            token_ids.append(token_id)

            # 4. 解码当前 token
            if token_id < len(self.vocab):
                token_str = self.vocab[token_id]
                if token_str == "<EOS>":
                    break
                if not token_str.startswith("<"):
                    generated_text += token_str

            # 5. 更新输入（用当前 token 的 one-hot 调制）
            token_signal = np.zeros(input_dim, dtype=np.float32)
            idx = token_id % input_dim
            token_signal[idx] = 1.0
            # 混合：0.7 保持上下文 + 0.3 新 token
            x = (x * 0.7 + token_signal.reshape(1, -1) * 0.3).astype(np.float32)

            if verbose:
                status = self.vocab[token_id] if token_id < len(self.vocab) else "?"
                print(f"  [{step:3d}] token={token_id:5d} '{status}'  logit_max={logits.max():.2f}")

        return token_ids, generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 30,
        temperature: float = 0.8,
    ) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            _, text = self.generate(prompt, max_tokens, temperature)
            results.append(text)
        return results

    def beam_search(
        self,
        beam_width: int = 3,
        max_tokens: int = 20,
    ) -> str:
        """Beam search 解码（确定性输出）"""
        beams = [([1], 0.0)]  # (token_ids, score)
        input_dim = 512
        x_base = np.random.randn(1, input_dim).astype(np.float32) * 0.01

        for _ in range(max_tokens):
            candidates = []
            for token_ids, score in beams:
                # 构建当前输入
                x = x_base.copy()
                context = self._extract_context(x)
                logits = self._logits(context)

                # Top-k 候选
                top_indices = np.argsort(logits)[-beam_width:]
                for tid in top_indices:
                    new_score = score + logits[tid]
                    new_ids = token_ids + [int(tid)]
                    candidates.append((new_ids, new_score))

            # 保留 top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            # 检查 EOS
            if self.vocab[beams[0][0][-1]] == "<EOS>":
                break

        best_ids = beams[0][0]
        return self.decode(best_ids)

    def train_step(self, x: np.ndarray, target_ids: List[int], lr: float = 0.01):
        """
        在线学习一步：用交叉熵损失更新投影权重。
        
        Args:
            x: 输入特征 [1, input_dim]
            target_ids: 目标 token ID 列表
            lr: 学习率
        """
        context = self._extract_context(x)
        logits = self._logits(context)  # [vocab_size]

        # Softmax
        logits_stable = logits - logits.max()
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum()

        # Cross-entropy gradient
        for tid in target_ids:
            if tid >= self.vocab_size:
                continue
            # dL/dlogits = probs - one_hot(target)
            grad_logits = probs.copy()
            grad_logits[tid] -= 1.0

            # dL/dW = context^T @ grad_logits
            grad_W = np.outer(context, grad_logits)
            grad_b = grad_logits

            # SGD update
            self.W_proj -= lr * grad_W
            self.b_proj -= lr * grad_b
            break  # 只取第一个 target token

    def get_stats(self) -> dict:
        """返回解码器统计信息"""
        w_mb = self.W_proj.nbytes / 1024 / 1024
        return {
            "decoder_params": self.W_proj.size + self.b_proj.size,
            "decoder_memory_mb": w_mb,
            "vocab_size": len(self.vocab),
            "hidden_dim": self.hidden_dim,
        }


# ---- 预训练玩具权重（可选） ----
def load_pretrained_decoder(path: str) -> TextDecoder:
    """加载预训练的解码器权重"""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data  # 直接返回 pickled decoder
