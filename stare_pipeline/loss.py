from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DEVICE, SimpleEncoderDecoder

MSE_WEIGHT = 0.1


class DynamicMarginContrastiveLoss(nn.Module):
    def forward(self, q, kp, kn, delta_s):
        q, kp, kn = (F.normalize(t.float(), dim=-1) for t in (q, kp, kn))
        sim_pos = (q * kp).sum(dim=-1)
        sim_neg = (q * kn).sum(dim=-1)
        return torch.clamp(sim_neg - sim_pos + delta_s, min=0.0).mean()


class CompressionAwareDynamicMarginLoss(nn.Module):
    def __init__(self, encoder_decoder: SimpleEncoderDecoder, mse_weight: float = MSE_WEIGHT):
        super().__init__()
        self.margin_loss = DynamicMarginContrastiveLoss()
        self.ed = encoder_decoder
        self.mse_weight = mse_weight

    def forward(self, q_feat, kp_feat, kn_feat, delta_s):
        margin = self.margin_loss(q_feat["sentence_embedding"], kp_feat["sentence_embedding"],
                                  kn_feat["sentence_embedding"], delta_s)
        mse = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        count = 0
        for feat in (q_feat, kp_feat, kn_feat):
            if "sentence_embedding_before_compress" in feat and "sentence_embedding_compressed" in feat:
                teacher = feat["sentence_embedding_before_compress"].float()
                student = self.ed.decode(feat["sentence_embedding_compressed"].float())
                mse = mse + F.mse_loss(student, teacher)
                count += 1
        if count > 0:
            mse = mse / count
        total = margin + self.mse_weight * mse
        return total, margin.item(), mse.item()
