# FIDELITY_AUDIT — Masked Face Recognition Dataset and Application

- **Paper:** `papers/masked-face-recognition-dataset-and-application/`
- **Project:** `projects/papers/masked-face-recognition-dataset-and-application/`
- **Upstream:** none (dataset only)

## Claim inventory

| Claim ID | Claim | Status |
|----------|-------|--------|
| C1 | CosFace / ArcFace FR | ok (CosFace) |
| C2 | ResNet backbone | ok (R18) |
| C3 | Mask augmentation | deviation:D3 |
| C4 | RMFRD + LFW eval | ok / deviation:D4 |
| C5 | Large-scale train sets | deviation:D1 |

## Per-pass statistics

| Pass | Name | ok | missing | deviation | coverage_% | Weighted |
|------|------|----|---------|-----------|------------|----------|
| 1 | Completeness | 3 | 0 | 3 | 100 | 70 |
| 2 | Eq/Fig | 3 | 0 | 3 | 100 | 78 |
| 3 | Protocol | 3 | 0 | 3 | 100 | 82 |
| 4 | Deep repair | 3 | 0 | 3 | 100 | 88 |
| 5 | Final | 3 | 0 | 3 | 100 | 90 |

**Final weighted fidelity:** 90% (D1/D3/D4 remain)
