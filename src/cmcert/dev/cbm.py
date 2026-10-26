"""A compact GPU-smoke-test CBM, separate from the preregistered architecture."""

from __future__ import annotations

import torch
from torch import nn


class DevCBM(nn.Module):
    def __init__(self, n_concepts: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.concept_head = nn.Linear(64, n_concepts)
        self.label_head = nn.Linear(n_concepts, 1)

    def forward(self, image: torch.Tensor):
        logits = self.concept_head(self.encoder(image))
        return logits, self.label_head(logits).squeeze(-1)
