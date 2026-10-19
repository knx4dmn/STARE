from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .loss import MSE_WEIGHT, CompressionAwareDynamicMarginLoss
from .model import BASE_MODEL, DEVICE, build_compression_stack, build_model

PHASE1_EPOCHS = 30
PHASE1_LR     = 1e-3
PHASE2_EPOCHS = 100
PHASE2_LR     = 2e-4
BATCH_SIZE    = 32
SEED          = 42


class TripletDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r["query"], r["positive_key"], r["negative_key"], float(r["delta_s"])


def _collate(batch):
    q, kp, kn, ds = zip(*batch)
    return list(q), list(kp), list(kn), torch.tensor(ds, dtype=torch.float32)


def _encode_with_grad(model, texts: list[str]) -> dict:
    features = model.tokenize(texts)
    features = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    for module in model:
        features = module(features)
    return features


def _encode_frozen(texts: list[str], base_model_name: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(base_model_name, device=DEVICE).eval()
    with torch.no_grad():
        emb = encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False).to(DEVICE).clone()
    del encoder
    return emb


def run_phase1(calibration_texts: list[str], kb_descriptions: list[str],
               base_model_name: str = BASE_MODEL) -> dict:
    calib_emb = _encode_frozen(calibration_texts, base_model_name)
    compress, dict_norm, ltq = build_compression_stack()
    with torch.no_grad():
        ltq.init_from_data(dict_norm(compress.encode(calib_emb)))

    params = list(compress.parameters()) + list(dict_norm.parameters()) + list(ltq.parameters())
    opt = torch.optim.AdamW(params, lr=PHASE1_LR)
    compress.train(); dict_norm.train(); ltq.train()
    for _ in range(PHASE1_EPOCHS):
        opt.zero_grad()
        quantized = ltq(dict_norm(compress.encode(calib_emb)))
        F.mse_loss(compress.decode(quantized), calib_emb).backward()
        opt.step()

    kb_emb = _encode_frozen(kb_descriptions, base_model_name)
    with torch.no_grad():
        ltq.init_from_data(dict_norm(compress.encode(kb_emb)))

    return {"compress": compress.state_dict(), "dict_norm": dict_norm.state_dict(), "ltq": ltq.state_dict()}


def run_phase2(model, phase1_state: dict, triplets: list[dict]):
    compress, dict_norm, ltq = build_compression_stack()
    compress.load_state_dict(phase1_state["compress"])
    dict_norm.load_state_dict(phase1_state["dict_norm"])
    ltq.load_state_dict(phase1_state["ltq"])
    model.add_module("compress", compress)
    model.add_module("dict_norm", dict_norm)
    model.add_module("ltq", ltq)
    model.to(DEVICE)

    loader = DataLoader(TripletDataset(triplets), batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)
    loss_fn = CompressionAwareDynamicMarginLoss(compress, mse_weight=MSE_WEIGHT)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=PHASE2_LR)

    model.train()
    for _ in range(PHASE2_EPOCHS):
        for queries, pos_keys, neg_keys, delta_s in loader:
            opt.zero_grad()
            loss, _, _ = loss_fn(_encode_with_grad(model, queries), _encode_with_grad(model, pos_keys),
                                 _encode_with_grad(model, neg_keys), delta_s.to(DEVICE))
            loss.backward()
            opt.step()
    return compress, dict_norm, ltq


def save_weights(model, compress, dict_norm, ltq, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, module in model._modules.items():
        if hasattr(module, "auto_model"):
            module.auto_model.save_pretrained(str(output_dir / "lora_adapter"))
            break
    torch.save({"compress": compress.state_dict(), "dict_norm": dict_norm.state_dict(),
                "ltq": ltq.state_dict()}, output_dir / "compression_stack.pt")


def kb_from_triplets(triplets: list[dict]) -> tuple[list[str], list[str]]:
    seen: dict[str, str] = {}
    for r in triplets:
        seen.setdefault(r["positive_method"], r["positive_key"])
        seen.setdefault(r["negative_method"], r["negative_key"])
    methods = sorted(seen)
    return methods, [seen[m] for m in methods]


def train(calibration_texts: list[str], triplets: list[dict], output_dir: Path,
          base_model_name: str = BASE_MODEL) -> None:
    torch.manual_seed(SEED)
    _, kb_descriptions = kb_from_triplets(triplets)
    phase1_state = run_phase1(calibration_texts, kb_descriptions, base_model_name)
    model = build_model(base_model_name)
    compress, dict_norm, ltq = run_phase2(model, phase1_state, triplets)
    save_weights(model, compress, dict_norm, ltq, output_dir)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="STARE two-phase training")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    args = parser.parse_args()
    calibration_texts = [r["calibration_text"] for r in _load_jsonl(args.calibration)]
    train(calibration_texts, _load_jsonl(args.triplets), args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
