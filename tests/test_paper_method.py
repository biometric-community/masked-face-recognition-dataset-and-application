"""Mask augmentation, the CosFace head and the verification error rate, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from maskfr.aug.mask_overlay import apply_mask_aug
from maskfr.eval import best_acc_eer
from maskfr.models.fr_net import CosFace, FaceEmbedder


def test_mask_covers_the_lower_face_and_keeps_the_eyes():
    """Sec. III: masks hide the lower face while the eyes and eyebrows stay visible."""
    x = torch.rand(4, 3, 112, 112) * 2 - 1
    out = apply_mask_aug(x, prob=1.0)
    y0 = int(0.52 * 112)
    assert torch.equal(out[:, :, :y0], x[:, :, :y0])
    assert (out[:, :, y0:] != x[:, :, y0:]).all()


def test_augmentation_probability_is_respected():
    x = torch.rand(4, 3, 112, 112)
    assert torch.equal(apply_mask_aug(x, prob=0.0), x)
    torch.manual_seed(0)
    masked = (apply_mask_aug(torch.rand(2000, 3, 8, 8), prob=0.5)[:, 0, 7, 0] == 0.85 * 2 - 1).float().mean()
    assert masked.item() == pytest.approx(0.5, abs=0.05)


def test_cosface_subtracts_the_margin_from_the_target_class_only():
    head = CosFace(2, 3, s=10.0, m=0.35)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3, 2))
    emb = torch.tensor([[1.0, 0.0]])
    logits = head(emb, torch.tensor([0]))
    cosine = F.linear(F.normalize(emb), F.normalize(head.weight))
    expected = 10.0 * cosine
    expected[0, 0] -= 10.0 * 0.35
    assert torch.allclose(logits, expected, atol=1e-6)


def test_embeddings_are_unit_length():
    model = FaceEmbedder(5, emb_dim=16, pretrained=False).eval()
    with torch.no_grad():
        emb = model.embed(torch.rand(3, 3, 112, 112))
    assert torch.allclose(emb.norm(dim=1), torch.ones(3), atol=1e-5)


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
