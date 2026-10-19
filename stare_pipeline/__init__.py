"""
STARE: lightweight statistical tool selection for SLM-based edge agents.

    profiler.py  — single-pass data profiler and profile-to-query template
    tools.py     — the 10 candidate statistical tools
    sandbox.py   — sandbox evaluator: fitness ∈ [0, 1] for every candidate tool
    triplets.py  — training triplets (query, k+, k−, Δs)
    model.py     — LoRA encoder + projection + LayerNorm + 2-bit LTQ
    loss.py      — utility-aware dynamic-margin loss + MSE regularizer
    train.py     — two-phase training
    router.py    — dot-product routing over the tool knowledge base

"""
