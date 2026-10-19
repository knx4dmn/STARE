from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .model import BASE_MODEL, DEVICE, build_compression_stack
from .profiler import format_query, profile_stream


def load_model(weights_dir: Path, base_model_name: str = BASE_MODEL):
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    weights_dir = Path(weights_dir)
    model = SentenceTransformer(base_model_name, device=DEVICE)
    for _, module in model._modules.items():
        if hasattr(module, "auto_model"):
            module.auto_model = PeftModel.from_pretrained(
                module.auto_model, str(weights_dir / "lora_adapter"), is_trainable=False).to(DEVICE)
            break

    ckpt = torch.load(weights_dir / "compression_stack.pt", map_location=DEVICE)
    compress, dict_norm, ltq = build_compression_stack()
    compress.load_state_dict(ckpt["compress"])
    dict_norm.load_state_dict(ckpt["dict_norm"])
    ltq.load_state_dict(ckpt["ltq"])
    model.add_module("compress", compress.eval())
    model.add_module("dict_norm", dict_norm.eval())
    model.add_module("ltq", ltq.eval())
    return model.eval()


class StareRouter:
    def __init__(self, model, kb_methods: list[str], kb_descriptions: list[str]):
        self.model = model
        self.kb_methods = kb_methods
        with torch.no_grad():
            self.kb_emb = model.encode(kb_descriptions, convert_to_tensor=True,
                                       show_progress_bar=False).float()

    def route(self, query: str) -> tuple[str, torch.Tensor]:
        with torch.no_grad():
            e_q = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False).float()
        scores = e_q @ self.kb_emb.T
        return self.kb_methods[int(scores.argmax().item())], scores.squeeze(0)

    def route_stream(self, series: pd.Series, task_category: str) -> str:
        return self.route(format_query(profile_stream(series), task_category))[0]
