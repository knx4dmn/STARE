# STARE

Lightweight statistical tool selection for SLM-based edge agents.

STARE profiles each sensor stream with a single-pass statistical profiler, scores candidate
statistical tools in a sandbox, and trains a compact embedding model with a utility-aware
dynamic-margin loss. At inference time, a tool is selected by dot-product retrieval over a
2-bit quantized knowledge base.

## Structure

```
stare_pipeline/
├── profiler.py   # single-pass profiler + profile-to-query template
├── tools.py      # 10 candidate statistical tools
├── sandbox.py    # sandbox evaluator (fitness in [0, 1])
├── triplets.py   # training triplets (query, k+, k-, Δs)
├── model.py      # LoRA encoder + projection + LayerNorm + 2-bit LTQ
├── loss.py       # dynamic-margin contrastive loss + MSE regularizer
├── train.py      # two-phase training
└── router.py     # dot-product routing
```

## Installation

```bash
pip install -r requirements.txt   # requires sentence-transformers>=3.3.0
```


## License

MIT
