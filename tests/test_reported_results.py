"""REPORT.md, the committed output of the full run, against the paper's results."""

from __future__ import annotations

import re


def row(report: str, dataset: str) -> dict:
    match = re.search(
        rf"\| {dataset} \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \| (\d+) \|", report
    )
    assert match, f"{dataset} row missing from REPORT.md"
    acc, eer, tpr2, tpr3, pairs = match.groups()
    return {"accuracy": float(acc), "eer": float(eer), "pairs": int(pairs)}


def test_masked_accuracy_reaches_the_published_level(report, targets):
    """Abstract: 95% accuracy on masked faces."""
    assert row(report, "rmfrd")["accuracy"] >= targets["masked_accuracy"] - targets["tolerance_pp"]


def test_masked_accuracy_improves_on_the_starting_point(report, targets):
    """Sec. III: masked-face accuracy rises from an initial 50%."""
    assert row(report, "rmfrd")["accuracy"] > targets["masked_accuracy_before"]


def test_unmasked_accuracy_stays_at_the_usual_level(report, targets):
    """Sec. IV: regular face recognition exceeds 99%."""
    assert row(report, "lfw")["accuracy"] >= targets["unmasked_accuracy"] - targets["tolerance_pp"]


def test_masked_pairs_are_harder_than_unmasked_pairs(report):
    assert row(report, "rmfrd")["accuracy"] < row(report, "lfw")["accuracy"]


def test_rmfrd_uses_the_official_verification_pairs(report, targets):
    assert row(report, "rmfrd")["pairs"] == targets["rmfrd_pairs"]
