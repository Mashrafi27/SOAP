from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ISAB, PMA, SAB


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super().__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output),
        )

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X))


class SetTransformerRegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 128,
        num_inds: int = 32,
        dim_hidden: int = 256,
        num_heads: int = 4,
        predictor_hidden: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.set_transformer = SetTransformer(
            dim_input=feature_dim,
            num_outputs=1,
            dim_output=latent_dim,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=True,
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, x, lengths):
        embeddings = []
        for i in range(x.size(0)):
            length = int(lengths[i])
            features = x[i, :length].unsqueeze(0)
            z = self.set_transformer(features).squeeze(0).squeeze(0)
            embeddings.append(z)
        embeddings = F.normalize(torch.stack(embeddings), dim=-1)
        preds = self.predictor(embeddings).squeeze(-1)
        return preds, embeddings


class SetTransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 510,
        num_inds: int = 32,
        dim_hidden: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.set_transformer = SetTransformer(
            dim_input=feature_dim,
            num_outputs=1,
            dim_output=latent_dim,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=True,
        )

    def forward(self, x, lengths):
        embeddings = []
        for i in range(x.size(0)):
            length = int(lengths[i])
            features = x[i, :length].unsqueeze(0)
            z = self.set_transformer(features).squeeze(0).squeeze(0)
            embeddings.append(z)
        return F.normalize(torch.stack(embeddings), dim=-1)
