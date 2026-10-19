from __future__ import annotations

import torch
import torch.nn as nn

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM      = 384
COMPRESSED_DIM = 256
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1
LORA_TARGETS   = ["query", "key", "value"]


class SimpleEncoderDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = nn.Linear(in_features, out_features)
        self.decoder = nn.Linear(out_features, in_features)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, features):
        is_dict = isinstance(features, dict)
        x = features["sentence_embedding"] if is_dict else features
        encoded = self.encode(x)
        if self.training and is_dict:
            features["sentence_embedding_before_compress"] = x.clone()
            features["sentence_embedding_compressed"] = encoded.clone()
        if is_dict:
            features["sentence_embedding"] = encoded
            return features
        return encoded


class DictLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, features):
        is_dict = isinstance(features, dict)
        x = features["sentence_embedding"] if is_dict else features
        x_norm = self.norm(x)
        if is_dict:
            features["sentence_embedding"] = x_norm
            return features
        return x_norm


class LTQ2BitQuantizer(nn.Module):
    def __init__(self, dim: int | None = None, init_range: float = 2.0, output_discrete: bool = False):
        super().__init__()
        self.per_dim = dim is not None
        C = dim if self.per_dim else 1
        self.num_levels = 4
        self.n_val = self.num_levels - 1
        self.output_discrete = output_discrete
        self.init_range = init_range
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.zeros(C))
        self.a = nn.Parameter(torch.ones(C, self.n_val) * self.interval)
        self.scale1 = nn.Parameter(torch.ones(C))
        self.scale2 = nn.Parameter(torch.ones(C))
        self.register_buffer("eps", torch.tensor(1e-3))

    @staticmethod
    def _compute_thresholds(start, a_pos):
        cumsum_a = torch.cumsum(a_pos, dim=-1)
        thre_backwards = start.unsqueeze(-1) + torch.cat(
            [torch.zeros_like(cumsum_a[:, :1]), cumsum_a[:, :-1]], dim=-1)
        thre_forwards = start.unsqueeze(-1) + cumsum_a - a_pos / 2
        return thre_forwards, thre_backwards

    def _quantize_levels(self, x, start, a_pos):
        start = start.float()
        a_pos = torch.clamp(a_pos.float(), min=self.eps.item())
        original_shape = x.shape
        if x.ndim == 0:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 1:
            x = x.unsqueeze(0)

        thre_fwd, thre_bwd = self._compute_thresholds(start, a_pos)
        thre_fwd, thre_bwd = thre_fwd.unsqueeze(0), thre_bwd.unsqueeze(0)
        x_forward = (x.unsqueeze(-1) > thre_fwd).float().sum(dim=-1)

        a = a_pos.unsqueeze(0)
        mask1 = ((x > thre_fwd[:, :, 0]) & (x <= thre_fwd[:, :, 1])).float()
        mask2 = ((x > thre_fwd[:, :, 1]) & (x <= thre_fwd[:, :, 2])).float()
        mask3 = (x > thre_fwd[:, :, 2]).float()
        x_backward = (
            mask1 * ((self.interval / a[:, :, 0]) * (x - thre_bwd[:, :, 0]) * mask1 + 0.0)
            + mask2 * ((self.interval / a[:, :, 1]) * (x - thre_bwd[:, :, 1]) * mask2 + 1.0)
            + mask3 * ((self.interval / a[:, :, 2]) * (x - thre_bwd[:, :, 2]) * mask3 + 2.0)
        )
        x_backward = torch.clamp(x_backward, min=0.0, max=3.0)
        return x_forward.reshape(original_shape), x_backward.reshape(original_shape)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor):
        q = torch.tensor([0.25, 0.50, 0.75], device=x.device, dtype=torch.float32)
        if self.per_dim:
            quantiles = torch.quantile(x.float(), q, dim=0).T
        else:
            quantiles = torch.quantile(x.float().view(-1), q).unsqueeze(0)
        self.start.data = quantiles[:, 0].to(self.start.dtype)
        self.a.data[:, 0] = (quantiles[:, 1] - quantiles[:, 0]).to(self.a.dtype)
        self.a.data[:, 1] = (quantiles[:, 2] - quantiles[:, 1]).to(self.a.dtype)
        self.a.data[:, 2] = ((quantiles[:, 2] - quantiles[:, 0]) / 2).to(self.a.dtype)

    def forward(self, x, return_hard: bool = False, return_discrete: bool = False, **kwargs):
        input_dict = x if isinstance(x, dict) else None
        x_tensor = x["sentence_embedding"] if input_dict is not None else x

        x_tensor = x_tensor * (self.scale1.float().unsqueeze(0) if self.per_dim else self.scale1.float())
        x_forward, x_backward = self._quantize_levels(x_tensor, self.start, self.a)
        if self.training and not return_hard:
            out = x_forward.detach() + x_backward - x_backward.detach()
        else:
            out = x_forward
        if not return_discrete and not (self.output_discrete and not self.training):
            out = out * (self.scale2.float().unsqueeze(0) if self.per_dim else self.scale2.float())

        if input_dict is not None:
            input_dict["sentence_embedding"] = out
            return input_dict
        return out


def build_model(base_model_name: str = BASE_MODEL):
    from peft import LoraConfig, TaskType
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(base_model_name, device=DEVICE)
    model.add_adapter(LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=LORA_TARGETS,
    ))
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.2%})")
    return model


def build_compression_stack():
    compress  = SimpleEncoderDecoder(EMBED_DIM, COMPRESSED_DIM).to(DEVICE)
    dict_norm = DictLayerNorm(COMPRESSED_DIM).to(DEVICE)
    ltq       = LTQ2BitQuantizer(dim=COMPRESSED_DIM, output_discrete=False).to(DEVICE)
    return compress, dict_norm, ltq
