from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.instance_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, M, F), lengths: (B,)
        h = self.instance_encoder(x)  # (B, M, H)
        attn = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        attn = self.attn_w(attn).squeeze(-1)  # (B, M)

        max_atoms = x.size(1)
        mask = torch.arange(max_atoms, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn = attn.masked_fill(~mask, -1e9)
        weights = torch.softmax(attn, dim=1).unsqueeze(-1)  # (B, M, 1)
        pooled = (weights * h).sum(dim=1)  # (B, H)
        pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)
