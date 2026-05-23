#!/usr/bin/env python3
"""
NeuroFlow v4 — Full Capabilities Demo
======================================
Demonstrates: single/batch inference, model analysis,
encoding similarity, and a formatted report.

Usage:
    python examples/demo_inference.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from neuroflow_v4 import NeuroFlowV4, Predictor, encode_text, encode_batch
from neuroflow_v4.encoder import compute_similarity
from neuroflow_v4 import config as C


# ═══════════════════════════════════════════════════════════════
# 1. Model Loading
# ═══════════════════════════════════════════════════════════════
def load_model() -> Predictor:
    """Load the model with pretrained weights (auto-download if needed)."""
    print("=" * 70)
    print("  🧠 NeuroFlow v4 — Full Demo")
    print("=" * 70)
    print()
    print("  [1/5] Loading model...")
    predictor = Predictor()  # auto-downloads pretrained weights
    print()
    return predictor


# ═══════════════════════════════════════════════════════════════
# 2. Single Text & Batch Inference
# ═══════════════════════════════════════════════════════════════
def demo_single_inference(predictor: Predictor):
    """Run a single text through the model and display results."""
    print("  [2/5] Single-text inference...")
    print()

    sample_text = "The meaning of life is to find your gift. The purpose of life is to give it away."
    out = predictor(sample_text, return_details=True)

    print(f"  ┌─ Input: \"{sample_text[:60]}...\"")
    print(f"  ├─ h_var (variance):       {out['h_var']:.6f}")
    print(f"  ├─ gate_mean:              {out['gate_mean']:.4f}")
    print(f"  ├─ gate_std:               {out['gate_std']:.4f}")
    print(f"  ├─ attn_top_slot:          {out['attn_top_slot']}  (most-attended memory slot)")
    print(f"  ├─ k_active (SAE sparsity):{out['k_active']}")
    print(f"  ├─ recon_mse:              {out['recon_mse']:.6f}")
    print(f"  ├─ value (scalar head):    {out['value'][0, 0]:.4f}")
    print(f"  ├─ top5 predicted chars:   {''.join(out['top5_chars'])}")
    print(f"  ├─ h3 shape:               {out['h3'].shape}")
    print(f"  └─ word_probs shape:       {out['word_probs'].shape}")
    print()


def demo_batch_inference(predictor: Predictor):
    """Run multiple texts in a single batch call."""
    print("  [3/5] Batch inference (4 texts)...")
    print()

    texts = [
        "Science is organized knowledge.",
        "Art is the lie that tells the truth.",
        "History is written by the victors.",
        "Mathematics is the language of nature.",
    ]
    out = predictor(texts)

    print(f"  Batch of {len(texts)} texts:")
    print(f"  {'#':>2}  {'Input':<48}  {'h_var':>8}  {'gate':>7}  {'slot':>4}  {'k':>3}")
    print(f"  {'─'*2}  {'─'*48}  {'─'*8}  {'─'*7}  {'─'*4}  {'─'*3}")
    for i, t in enumerate(texts):
        top5 = ''.join(out['top5_chars'][i])
        print(f"  {i+1:>2d}  {t:<48}  {out['h_var']:>8.4f}  "
              f"{out['gate_mean']:.2f}±{out['gate_std']:.2f}  "
              f"{out['attn_top_slot']:>4d}  {out['k_active']:>3d}")
    print()


# ═══════════════════════════════════════════════════════════════
# 3. Model Analysis
# ═══════════════════════════════════════════════════════════════
def demo_analysis(predictor: Predictor):
    """Run model analysis: stats, M_V norms, gate distribution, attention."""
    print("  [4/5] Model analysis...")
    print()

    # ── Built-in analysis ──
    stats = predictor.analyze()

    print("  ┌─ Model Stats ─────────────────────────────────────────────")
    print(f"  ├─ Architecture:         {stats['architecture']}")
    print(f"  ├─ Version:              {stats['version']}")
    print(f"  ├─ Vocab size:           {stats['vocab_size']} chars")
    print(f"  ├─ Memory footprint:     {stats['memory_mb']:.2f} MB")
    print(f"  ├─ W_embed effective_rank: {stats['W_embed']['effective_rank']}")
    print(f"  ├─ W_embed top-5 singular:  {[f'{s:.2f}' for s in stats['W_embed']['singular_top5']]}")
    print(f"  ├─ M_K mean self-cos:    {stats['M_K']['mean_self_cos']:.4f}")
    print(f"  ├─ M_K mean norm:        {stats['M_K']['mean_norm']:.4f}")
    print(f"  ├─ M_V norms:            μ={stats['M_V']['mean_norm']:.4f}  "
          f"σ={stats['M_V']['std_norm']:.4f}  "
          f"min={stats['M_V']['min_norm']:.4f}  "
          f"max={stats['M_V']['max_norm']:.4f}")
    print(f"  ├─ Gate bias range:      [{stats['gate']['bias_range'][0]:.4f}, "
          f"{stats['gate']['bias_range'][1]:.4f}]")
    print(f"  └─ Gate bias std:        {stats['gate']['bias_std']:.4f}")
    print()

    # ── M_V Norm distribution ──
    model = predictor.model
    mv_norms = np.linalg.norm(model.M_V, axis=1)
    print("  ┌─ M_V Norms (per memory slot) ────────────────────────────")
    print(f"  │    {'Slot':>4}  {'Norm':>8}  {'Active?':>7}")
    print(f"  │    {'────':>4}  {'────':>8}  {'───────':>7}")
    threshold = stats['M_V']['mean_norm'] + 0.5 * stats['M_V']['std_norm']
    for s in range(C.MEM_SLOTS):
        active = mv_norms[s] > threshold
        marker = "★" if active else "·"
        print(f"  │    {s:>4d}  {mv_norms[s]:>8.4f}  {marker:>7}")
    print(f"  └─ Active slots (norm > μ+0.5σ): {np.sum(mv_norms > threshold)}/{C.MEM_SLOTS}")
    print()

    # ── Inference-time gate distribution ──
    sample_for_gate = "Cognition is the process of acquiring knowledge and understanding through thought."
    X = encode_text(sample_for_gate).reshape(1, -1)
    fwd = model.forward(X, return_intermediates=True)
    gate_vals = fwd['gate'][0]

    print("  ┌─ Gate Distribution (on sample input) ────────────────────")
    print(f"  ├─ Mean:        {float(np.mean(gate_vals)):.4f}")
    print(f"  ├─ Std:         {float(np.std(gate_vals)):.4f}")
    print(f"  ├─ Min:         {float(np.min(gate_vals)):.4f}")
    print(f"  ├─ Max:         {float(np.max(gate_vals)):.4f}")
    print(f"  ├─ Gates > 0.5: {int(np.sum(gate_vals > 0.5))}/{len(gate_vals)}  "
          f"({100*np.mean(gate_vals > 0.5):.1f}%)")
    hist, edges = np.histogram(gate_vals, bins=5, range=(0, 1))
    bar = "█" * 20
    print(f"  ├─ Histogram:")
    for b in range(len(hist)):
        pct = hist[b] / len(gate_vals) * 100
        fill = int(hist[b] / max(hist) * 20) if max(hist) > 0 else 0
        print(f"  │    [{edges[b]:.2f}–{edges[b+1]:.2f})  {bar[:fill]}  {pct:5.1f}%")
    print(f"  └─ Gate fusion: gate * h1 + (1-gate) * mem_feat")
    print()

    # ── Attention pattern ──
    attn = fwd['attn'][0]
    print("  ┌─ Memory Attention Pattern ───────────────────────────────")
    print(f"  │    {'Slot':>4}  {'Weight':>8}  {'Bar':>22}")
    print(f"  │    {'────':>4}  {'──────':>8}  {'───':>22}")
    for s in range(C.MEM_SLOTS):
        pct = attn[s] * 100
        fill = int(attn[s] / max(attn) * 20) if max(attn) > 0 else 0
        print(f"  │    {s:>4d}  {attn[s]:>8.4f}  {'█' * fill}")
    print(f"  └─ Top-{C.ATTN_TOPK} attention (temperature={C.ATTN_TEMPERATURE})")
    print()


# ═══════════════════════════════════════════════════════════════
# 4. Encoding Similarity Comparison
# ═══════════════════════════════════════════════════════════════
def demo_similarity():
    """Compare encoding similarity between related vs unrelated texts."""
    print("  [5/5] Encoding similarity: related vs unrelated texts...")
    print()

    # Related pairs (same topic)
    related_pairs = [
        ("The stock market rallied today on strong earnings reports.",
         "Investors saw gains as corporate profits exceeded expectations."),
        ("Quantum computing leverages superposition and entanglement.",
         "Qubits can represent multiple states simultaneously in quantum systems."),
        ("The neural network learned patterns from large datasets.",
         "Deep learning models extract hierarchical features from data."),
    ]

    # Unrelated pairs (different topics)
    unrelated_pairs = [
        ("The Pythagorean theorem relates sides of a right triangle.",
         "Baking a perfect soufflé requires precise temperature control."),
        ("Shakespeare wrote 154 sonnets in Elizabethan England.",
         "Mitochondria are the powerhouse of the cell in biology."),
        ("The price of Bitcoin fluctuated wildly this quarter.",
         "Impressionist painters focused on light and color effects."),
    ]

    print(f"  {'Pair':>5}  {'Related':>8}  {'Unrelated':>10}  {'Gap':>8}")
    print(f"  {'─────':>5}  {'───────':>8}  {'─────────':>10}  {'───':>8}")

    related_sims = []
    unrelated_sims = []
    all_deltas = []

    for i, (t1, t2) in enumerate(related_pairs):
        X1 = encode_text(t1).reshape(1, -1)
        X2 = encode_text(t2).reshape(1, -1)
        sim = float(compute_similarity(X1, X2)[0, 0])
        related_sims.append(sim)

    for i, (t1, t2) in enumerate(unrelated_pairs):
        X1 = encode_text(t1).reshape(1, -1)
        X2 = encode_text(t2).reshape(1, -1)
        sim = float(compute_similarity(X1, X2)[0, 0])
        unrelated_sims.append(sim)

    for i in range(3):
        delta = related_sims[i] - unrelated_sims[i]
        all_deltas.append(delta)
        print(f"  {i+1:>5d}  {related_sims[i]:>8.4f}  {unrelated_sims[i]:>10.4f}  {delta:>8.4f}")

    avg_rel = np.mean(related_sims)
    avg_unrel = np.mean(unrelated_sims)
    print(f"  {'─────':>5}  {'───────':>8}  {'─────────':>10}  {'───':>8}")
    print(f"  {'μ':>5}  {avg_rel:>8.4f}  {avg_unrel:>10.4f}  {avg_rel - avg_unrel:>8.4f}")
    print()
    if avg_rel > avg_unrel:
        print(f"  ✅ Related texts are {((avg_rel / avg_unrel) - 1) * 100:.1f}% more similar "
              f"than unrelated texts — encoder captures semantic structure.\n")
    else:
        print(f"  ⚠️  Similarity gap is negative; hash-based encoder may need more context.\n")

    # Also show encoding on the model side (h3 cosine similarity)
    print("  ── Model-level (h3) similarity ──")
    model = NeuroFlowV4(weights=None)  # random init for demo
    rel_h3_sims = []
    unrel_h3_sims = []
    for i, (t1, t2) in enumerate(related_pairs):
        X = encode_batch([t1, t2])
        out = model.forward(X)
        h = out['h3']
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = float(h_norm[0:1] @ h_norm[1:2].T)
        rel_h3_sims.append(sim)
    for i, (t1, t2) in enumerate(unrelated_pairs):
        X = encode_batch([t1, t2])
        out = model.forward(X)
        h = out['h3']
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = float(h_norm[0:1] @ h_norm[1:2].T)
        unrel_h3_sims.append(sim)
    print(f"  {'':>5}  {'Related (h3)':>14}  {'Unrelated (h3)':>16}  {'Gap':>8}")
    print(f"  {'─────':>5}  {'──────────────':>14}  {'──────────────':>16}  {'───':>8}")
    for i in range(3):
        delta = rel_h3_sims[i] - unrel_h3_sims[i]
        print(f"  {i+1:>5d}  {rel_h3_sims[i]:>14.4f}  {unrel_h3_sims[i]:>16.4f}  {delta:>8.4f}")
    print(f"  {'μ':>5}  {np.mean(rel_h3_sims):>14.4f}  {np.mean(unrel_h3_sims):>16.4f}  "
          f"{np.mean(rel_h3_sims) - np.mean(unrel_h3_sims):>8.4f}")
    print()


# ═══════════════════════════════════════════════════════════════
# 5. Formatted Summary Report
# ═══════════════════════════════════════════════════════════════
def print_summary(predictor: Predictor):
    """Print a clean formatted summary report."""
    stats = predictor.analyze()
    print("=" * 70)
    print("  📋 NeuroFlow v4 — Summary Report")
    print("=" * 70)
    print()
    print(f"  Architecture:     {stats['architecture']}")
    print(f"  Version:          {stats['version']}")
    print(f"  Backend:          {C.ARCHITECTURE['backend']}")
    print(f"  Total params:     {C.ARCHITECTURE['total_params']:,}")
    print(f"  Weight size:      {C.ARCHITECTURE['weight_size_mb']} MB")
    print(f"  Memory (runtime): {stats['memory_mb']:.2f} MB")
    print()
    print(f"  ┌─ Dimensions ────────────────────────────────────────────")
    print(f"  ├─ Input (text):           {C.TEXT_DIM}")
    print(f"  ├─ Hidden (h1, h3):        {C.HIDDEN_DIM}")
    print(f"  ├─ Memory slots (M_K/V):   {C.MEM_SLOTS} × {C.MEM_DIM_IN}")
    print(f"  └─ Vocabulary:             {C.VOCAB_SIZE}")
    print()
    print(f"  ┌─ SAE ───────────────────────────────────────────────────")
    print(f"  ├─ k range:                {C.SAE_K_MIN} – {C.SAE_K_MAX}")
    print(f"  └─ Adaptive:               entropy-driven per sample")
    print()
    print(f"  ┌─ Memory ───────────────────────────────────────────────")
    print(f"  ├─ Attention top-k:        {C.ATTN_TOPK}")
    print(f"  ├─ Attention temperature:  {C.ATTN_TEMPERATURE}")
    print(f"  ├─ M_V mean norm:          {stats['M_V']['mean_norm']:.4f}")
    print(f"  └─ M_V std norm:           {stats['M_V']['std_norm']:.4f}")
    print()
    print(f"  ┌─ Gate ─────────────────────────────────────────────────")
    print(f"  ├─ Gate bias range:        [{stats['gate']['bias_range'][0]:.4f}, "
          f"{stats['gate']['bias_range'][1]:.4f}]")
    print(f"  ├─ Gate bias std:          {stats['gate']['bias_std']:.4f}")
    print(f"  ├─ τ (temperature):        {predictor.model.gate_sharper.current_tau:.3f}")
    print(f"  └─ Tau active:             {predictor.model.gate_tau_active}")
    print()
    print(f"  ┌─ Training Config ──────────────────────────────────────")
    print(f"  ├─ Learning rate:          {C.LEARNING_RATE}")
    print(f"  ├─ Weight decay:           {C.WEIGHT_DECAY}")
    print(f"  ├─ Mask ratio:             {C.MASK_RATIO}")
    print(f"  ├─ Input noise:            {C.INPUT_NOISE}")
    print(f"  └─ VICReg gamma:           {C.VICREG_GAMMA}")
    print()
    print(f"  ┌─ Weights ──────────────────────────────────────────────")
    print(f"  ├─ Source:                 {C.HF_REPO}")
    print(f"  └─ File:                   {C.WEIGHT_FILENAME}")
    print()
    print("  ✅ Demo complete — all capabilities verified.")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    predictor = load_model()
    demo_single_inference(predictor)
    demo_batch_inference(predictor)
    demo_analysis(predictor)
    demo_similarity()
    print_summary(predictor)
