# shield_can/model.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

from .config import ModelConfig


class EdgeTransformer(nn.Module):
    """
    Encoder-only Transformer for SHIELD-CAN-style features.
    Input: (batch, window, feature_dim)
    Output: logits over attack classes per window.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = cfg.feature_dim
        self.d_model = cfg.d_model
        self.window = cfg.window_size

        self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        self.stat_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.window + 1, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.cls_head = nn.Linear(self.d_model, cfg.num_classes)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.stat_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.xavier_uniform_(self.cls_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, F)
        Returns logits: (B, num_classes)
        """
        B, W, F = x.shape
        assert W == self.window, f"expected window {self.window}, got {W}"
        assert F == self.feature_dim, f"expected feature_dim {self.feature_dim}, got {F}"

        x = self.input_proj(x) * math.sqrt(self.d_model)

        stat = self.stat_token.expand(B, 1, -1)
        x = torch.cat([stat, x], dim=1)  # (B, W+1, d_model)

        x = x + self.pos_embedding[:, : W + 1, :]

        h = self.encoder(x)
        h_stat = h[:, 0, :]  # STAT token
        logits = self.cls_head(h_stat)
        return logits
