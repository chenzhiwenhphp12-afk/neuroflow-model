#!/usr/bin/env python3
"""NeuroFlow v4 — Quick Start Demo"""
# pip install neuroflow-v4  (or: git clone + pip install -e .)
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neuroflow_v4 import Predictor
predictor = Predictor(weights_path="random")
texts = [
    ("Science", "The universe is under no obligation to make sense to you."),
    ("Literature", "It was the best of times, it was the worst of times."),
    ("Philosophy", "I think therefore I am."),
]
for label, t in texts:
    out = predictor(t)
    print(f"[{label}] h_var={out['h_var']:.4f}  gate={out['gate_mean']:.3f}\u00b1{out['gate_std']:.3f}")
    print(f"  top5={''.join(out['top5_chars'])}\n")
