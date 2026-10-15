"""ResNet-18 + CosFace head (paper benchmarks CosFace/ArcFace + ResNet)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CosFace(nn.Module):
    """Large-margin cosine loss (CosFace / AM-Softmax)."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (torch.where(one_hot.bool(), phi, cosine))
        return logits


class FaceEmbedder(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 512, pretrained: bool = True, loss: str = "cosface"):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.loss_name = loss
        if loss == "cosface":
            self.head = CosFace(emb_dim, n_classes)
        else:
            self.head = nn.Linear(emb_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        if self.loss_name == "cosface":
            if labels is None:
                raise ValueError("CosFace head requires labels")
            return self.head(emb, labels), emb
        return self.head(emb), emb
