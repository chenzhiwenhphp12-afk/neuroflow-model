#!/usr/bin/env python3 -u
"""
NeuroFlow 蒸馏训练 — 用教师生成的数据训练学生模型

流程:
  1. 加载教师数据 (JSONL: {"prompt":..., "completion":...} 或纯文本)
  2. 加载 NeuroFlow 学生模型 (NF + LM head)
  3. 用 Cross-Entropy 训练学生预测教师文本 (滑动窗口多token预测)
  4. 保存 checkpoint (LMH1/LMH2 兼容格式)

用法:
  python3 -u scripts/train_distill.py \
    --teacher-data teacher_data.jsonl \
    --nf-model checkpoint/model.nfv1 \
    --lm-model checkpoint/lm_head.nfv1 \
    --tokenizer configs/tokenizer_128k.json \
    --output ./distill_output \
    --epochs 5 --lr 5e-6 --batch-size 32 \
    --train-nf   # 可选: 同时训练NF权重

注意: 必须使用 python3 -u 运行，或在后台运行时设置 PYTHONUNBUFFERED=1
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse, json, struct, sys, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_full import load_nfv1, load_lmh1, load_tokenizer, encode, layernorm, gelu, softmax


def forward_with_cache(token_ids, nf_w, lm_w, vocab_size=128000):
    d_model = nf_w['input_proj.weight'].shape[1]
    hidden_dim = nf_w['input_proj.weight'].shape[0]
    cache = {}

    x = np.zeros(d_model, dtype=np.float32)
    copy_len = min(len(token_ids), d_model)
    for j in range(copy_len):
        x[j] = float(token_ids[j]) / float(vocab_size)

    h = nf_w['input_proj.weight'] @ x + nf_w['input_proj.bias']
    cache['input_proj.x'] = x.copy()
    cache['input_proj.pre_norm'] = h.copy()
    h = layernorm(h, nf_w['input_proj_norm.weight'], nf_w['input_proj_norm.bias'])
    cache['input_proj.post_norm'] = h.copy()
    h = gelu(h)
    cache['input_proj.post_gelu'] = h.copy()

    g1 = gelu(nf_w['sn.gate1.weight'] @ h + nf_w['sn.gate1.bias'])
    gates = softmax(nf_w['sn.gate2.weight'] @ g1 + nf_w['sn.gate2.bias'])
    cache['gates'] = gates.copy()

    h_ecn = h.copy()
    cache['ecn.h0'] = h_ecn.copy()
    for i in range(12):
        h_ecn = gelu(nf_w[f'ecn.dlpfc{i}.weight'] @ h_ecn + nf_w[f'ecn.dlpfc{i}.bias'])
    cache['ecn.last'] = h_ecn.copy()

    vmpfc = gelu(nf_w['ecn.vmpfc1.weight'] @ h_ecn + nf_w['ecn.vmpfc1.bias'])
    decision = nf_w['ecn.vmpfc2.weight'] @ vmpfc + nf_w['ecn.vmpfc2.bias']
    cache['decision'] = decision.copy()

    mem_encoded = nf_w['memory.encode.weight'] @ h + nf_w['memory.encode.bias']
    cache['mem_encoded'] = mem_encoded.copy()

    dmn_enc = gelu(nf_w['dmn.mem_encoder1.weight'] @ mem_encoded + nf_w['dmn.mem_encoder1.bias'])
    dmn_latent = nf_w['dmn.mem_encoder2.weight'] @ dmn_enc + nf_w['dmn.mem_encoder2.bias']
    cache['dmn_latent'] = dmn_latent.copy()

    assoc_outs = []
    for i in range(8):
        a1 = gelu(nf_w[f'dmn.head{i}.1.weight'] @ dmn_latent + nf_w[f'dmn.head{i}.1.bias'])
        a2 = nf_w[f'dmn.head{i}.2.weight'] @ a1 + nf_w[f'dmn.head{i}.2.bias']
        assoc_outs.append(a2)
    dmn_vision = gelu(nf_w['dmn.future_proj1.weight'] @ np.concatenate(assoc_outs) + nf_w['dmn.future_proj1.bias'])
    cache['dmn_vision'] = dmn_vision.copy()

    mem_bank = nf_w['memory.bank']
    att = softmax(mem_encoded @ mem_bank.T)
    retrieved = att @ mem_bank
    mem_retrieved = nf_w['memory.retrieve.weight'] @ retrieved + nf_w['memory.retrieve.bias']
    cache['mem_retrieved'] = mem_retrieved.copy()

    ecn_w = decision * gates[0]
    dmn_w = dmn_vision * gates[1]
    dmn_w_pad = np.zeros(hidden_dim, dtype=np.float32)
    dmn_w_pad[:dmn_w.shape[0]] = dmn_w
    mem_w = np.zeros(hidden_dim, dtype=np.float32)
    mem_w[:mem_retrieved.shape[0]] = mem_retrieved
    combined = np.concatenate([ecn_w, dmn_w_pad, mem_w])
    cache['combined'] = combined.copy()

    fused = nf_w['output_fusion.down.weight'] @ combined + nf_w['output_fusion.down.bias']
    fused_pre_relu = layernorm(fused, nf_w['output_fusion.bn_norm.weight'], nf_w['output_fusion.bn_norm.bias'])
    cache['fusion.pre_relu'] = fused_pre_relu.copy()
    fused_relu = np.maximum(0, fused_pre_relu)
    cache['fusion.post_relu'] = fused_relu.copy()
    nf_output = nf_w['output_fusion.up.weight'] @ fused_relu + nf_w['output_fusion.up.bias']
    nf_output = layernorm(nf_output, nf_w['output_fusion.norm.weight'], nf_w['output_fusion.norm.bias'])
    cache['nf_output'] = nf_output.copy()

    bridge_h = lm_w['bridge.weight'] @ nf_output + lm_w['bridge.bias']
    cache['bridge_h'] = bridge_h.copy()

    projected = lm_w['w_proj.weight'] @ bridge_h + lm_w['w_proj.bias']
    cache['projected'] = projected.copy()

    logits = lm_w['w_embed'] @ projected
    cache['logits'] = logits.copy()

    return logits, cache



def distill_step(token_ids, nf_w, lm_w, lr, vocab_size=128000, grad_clip=4.0,
                 train_nf=False, max_predictions=0):
    """单步蒸馏训练: 滑动窗口多token预测 + 反向传播

    对序列中每个位置 t，用 token_ids[:t+1] 前向预测 token_ids[t+1]。
    NF模型是"序列→单向量"架构，无法像Transformer那样单次前向获取所有位置hidden states，
    因此每个位置需要独立前向。max_predictions 限制每样本预测位置数以控制性能开销。

    --train-nf 反向传播路径止于 output_fusion.up/down，未穿过 ECN/DMN/Memory/SN gate。
    设计为渐进式解冻: 先训练 LM head + bridge，再解冻 output_fusion，最后解冻更深层。
    """
    seq_len = len(token_ids)
    if seq_len < 2:
        return 0.0

    total_loss = 0.0
    accum_lm_grads = {}
    accum_nf_grads = {}
    num_preds = 0

    positions = list(range(seq_len - 1))
    if max_predictions > 0 and len(positions) > max_predictions:
        step = max(1, len(positions) // max_predictions)
        positions = positions[::step][:max_predictions]

    for t in positions:
        prefix = token_ids[:t + 1]
        target_id = token_ids[t + 1]
        if target_id >= vocab_size:
            continue

        logits, cache = forward_with_cache(prefix, nf_w, lm_w, vocab_size)

        max_val = logits.max()
        exp_vals = np.exp(logits - max_val)
        sum_exp = exp_vals.sum()
        probs = exp_vals / sum_exp

        p_target = max(probs[target_id], 1e-10)
        total_loss += -math.log(p_target)
        num_preds += 1

        grad_logits = probs.copy()
        grad_logits[target_id] -= 1.0

        grad_w_embed = np.outer(grad_logits, cache['projected'])
        grad_projected = lm_w['w_embed'].T @ grad_logits

        grad_w_proj_weight = np.outer(grad_projected, cache['bridge_h'])
        grad_w_proj_bias = grad_projected.copy()
        grad_bridge_h = lm_w['w_proj.weight'].T @ grad_projected

        grad_bridge_weight = np.outer(grad_bridge_h, cache['nf_output'])
        grad_bridge_bias = grad_bridge_h.copy()

        step_lm_grads = {
            'w_embed': grad_w_embed,
            'w_proj.weight': grad_w_proj_weight,
            'w_proj.bias': grad_w_proj_bias,
            'bridge.weight': grad_bridge_weight,
            'bridge.bias': grad_bridge_bias,
        }

        for name, grad in step_lm_grads.items():
            if name not in accum_lm_grads:
                accum_lm_grads[name] = np.zeros_like(lm_w[name])
            accum_lm_grads[name] += grad

        if train_nf:
            grad_nf_output = lm_w['bridge.weight'].T @ grad_bridge_h
            grad_fused_relu = nf_w['output_fusion.up.weight'].T @ grad_nf_output
            grad_fused_pre_relu = grad_fused_relu * (cache['fusion.pre_relu'] > 0).astype(np.float32)

            step_nf_grads = {
                'output_fusion.up.weight': np.outer(grad_nf_output, cache['fusion.post_relu']),
                'output_fusion.up.bias': grad_nf_output.copy(),
                'output_fusion.down.weight': np.outer(grad_fused_pre_relu, cache['combined']),
                'output_fusion.down.bias': grad_fused_pre_relu.copy(),
            }

            for name, grad in step_nf_grads.items():
                if name in nf_w and nf_w[name].shape == grad.shape:
                    if name not in accum_nf_grads:
                        accum_nf_grads[name] = np.zeros_like(nf_w[name])
                    accum_nf_grads[name] += grad

    if num_preds == 0:
        return 0.0

    total_loss /= num_preds

    all_grads = {}
    for name, grad in accum_lm_grads.items():
        all_grads[f'lm.{name}'] = grad / num_preds
    for name, grad in accum_nf_grads.items():
        all_grads[f'nf.{name}'] = grad / num_preds

    total_norm = 0.0
    for g in all_grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = math.sqrt(total_norm)

    clip_scale = 1.0
    if total_norm > grad_clip and grad_clip > 0:
        clip_scale = grad_clip / total_norm

    effective_lr = lr * clip_scale
    for name, grad in all_grads.items():
        if name.startswith('lm.'):
            key = name[3:]
            if key in lm_w and lm_w[key].shape == grad.shape:
                lm_w[key] -= effective_lr * grad
        elif name.startswith('nf.'):
            key = name[3:]
            if key in nf_w and nf_w[key].shape == grad.shape:
                nf_w[key] -= effective_lr * grad

    return total_loss


def save_lmh1(path, lm_w):
    with open(path, 'wb') as f:
        f.write(b'LMH1')
        for name, arr in lm_w.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(arr.shape)))
            for d in arr.shape:
                f.write(struct.pack('<I', d))
            data = arr.astype(np.float32).tobytes()
            f.write(struct.pack('<I', len(data)))
            f.write(data)
        f.write(struct.pack('<I', 0))


def save_nfv1(path, nf_w):
    with open(path, 'wb') as f:
        f.write(b'NFv1')
        for name, arr in nf_w.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(arr.shape)))
            for d in arr.shape:
                f.write(struct.pack('<I', d))
            data = arr.astype(np.float32).tobytes()
            f.write(struct.pack('<I', len(data)))
            f.write(data)
        f.write(struct.pack('<I', 0))


def main():
    parser = argparse.ArgumentParser(description='NeuroFlow Distillation Training')
    parser.add_argument('--teacher-data', required=True, help='教师数据 (JSONL/TXT)')
    parser.add_argument('--nf-model', required=True, help='学生 NF 模型路径')
    parser.add_argument('--lm-model', required=True, help='学生 LM head 路径')
    parser.add_argument('--tokenizer', required=True, help='分词器路径')
    parser.add_argument('--output', default='./distill_output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=4.0)
    parser.add_argument('--train-nf', action='store_true', help='同时训练NF权重(默认只训练LM head)')
    parser.add_argument('--resume', default='', help='断点续训: 指定checkpoint目录')
    parser.add_argument('--max-predictions', type=int, default=8, help='每样本最大预测位置数(0=全部, 默认8)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("加载教师数据...")
    samples = []
    with open(args.teacher_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get('prompt', '') + rec.get('completion', '')
            except json.JSONDecodeError:
                text = line
            if len(text) >= 10:
                samples.append(text)
    print(f"   {len(samples)} 个样本")

    print("加载分词器...")
    vocab, id2token, merge_ranks = load_tokenizer(args.tokenizer)
    print(f"   词表: {len(vocab)} tokens")

    print("分词...")
    tokenized = []
    total_tokens = 0
    t0 = time.time()
    progress_interval = max(1, len(samples) // 20)
    for si, text in enumerate(samples):
        ids = encode(text, vocab, merge_ranks, max_len=128)
        if len(ids) >= 4:
            tokenized.append(ids)
            total_tokens += len(ids)
        if (si + 1) % progress_interval == 0 or si == len(samples) - 1:
            elapsed = time.time() - t0
            pct = (si + 1) * 100 // len(samples)
            rate = (si + 1) / max(elapsed, 0.01)
            eta = (len(samples) - si - 1) / max(rate, 0.01)
            print(f"   分词进度: {si+1}/{len(samples)} ({pct}%) | "
                  f"{rate:.0f} samples/s | ETA {eta:.0f}s | "
                  f"tokens={total_tokens:,}")
    avg_len = total_tokens / max(len(tokenized), 1)
    elapsed = time.time() - t0
    print(f"   完成: {total_tokens:,} tokens ({avg_len:.0f} avg/sample) | 耗时 {elapsed:.1f}s")

    print("加载学生模型...")
    nf_w = load_nfv1(args.nf_model)
    lm_w = load_lmh1(args.lm_model)
    d_model = nf_w['input_proj.weight'].shape[1]
    hidden_dim = nf_w['input_proj.weight'].shape[0]
    print(f"   NF: {len(nf_w)}层 | LM: {len(lm_w)}层 | d_model={d_model} hidden={hidden_dim}")

    start_step = 0
    start_epoch = 0
    if args.resume:
        print(f"断点续训: {args.resume}")
        nf_w = load_nfv1(f"{args.resume}/model.nfv1")
        lm_w = load_lmh1(f"{args.resume}/lm_head.nfv1")
        state_path = f"{args.resume}/training_state.json"
        if os.path.exists(state_path):
            with open(state_path) as sf:
                state = json.load(sf)
            start_step = state.get('step', 0)
            start_epoch = state.get('epoch', 1) - 1
            print(f"   恢复: step={start_step}, epoch={start_epoch+1}")

    mode_str = "LM+NF" if args.train_nf else "LM only"
    print(f"\n开始蒸馏训练 ({args.epochs} epochs, lr={args.lr}, batch={args.batch_size}, "
          f"grad_clip={args.grad_clip}, mode={mode_str})")
    print(f"   每样本预测位置数: {args.max_predictions if args.max_predictions > 0 else int(avg_len)} (max_predictions={args.max_predictions})")
    print("=" * 60)

    global_step = start_step
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        steps = 0

        indices = list(range(len(tokenized)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i:i + args.batch_size]

            batch_loss = 0.0
            for idx in batch_indices:
                loss = distill_step(tokenized[idx], nf_w, lm_w, args.lr,
                                    vocab_size=len(vocab), grad_clip=args.grad_clip,
                                    train_nf=args.train_nf, max_predictions=args.max_predictions)
                batch_loss += loss

            batch_loss /= len(batch_indices)
            epoch_loss += batch_loss
            steps += 1
            global_step += 1

            if steps % 10 == 0:
                print(f"  [Epoch {epoch+1}][Step {global_step}] loss={batch_loss:.4f}")

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                ckpt_dir = f"{args.output}/step_{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                save_nfv1(f"{ckpt_dir}/model.nfv1", nf_w)
                save_lmh1(f"{ckpt_dir}/lm_head.nfv1", lm_w)
                state = {"step": global_step, "epoch": epoch + 1, "loss": batch_loss, "lr": args.lr}
                with open(f"{ckpt_dir}/training_state.json", 'w') as sf:
                    json.dump(state, sf, indent=2)
                print(f"  Checkpoint: {ckpt_dir}")

        avg_loss = epoch_loss / max(steps, 1)
        print(f"=== Epoch {epoch+1} 完成, avg_loss={avg_loss:.4f} ===\n")

    final_dir = f"{args.output}/final"
    os.makedirs(final_dir, exist_ok=True)
    save_nfv1(f"{final_dir}/model.nfv1", nf_w)
    save_lmh1(f"{final_dir}/lm_head.nfv1", lm_w)
    print(f"最终模型已保存: {final_dir}")
    print("蒸馏训练完成")


if __name__ == '__main__':
    main()
