# REPORT — Masked Face Recognition Dataset and Application

**Protocol:** `full`

Lab twin: CosFace ResNet-18 trained on CelebA with lower-face mask augmentation;
evaluated on LFW and RMFRD masked↔unmasked verification pairs.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Mask aug prob: 0.5
- Elapsed (s): 1056.8

![Training](outputs/figures/full/train_loss.svg)

## Verification

| Dataset | Accuracy | EER | TPR@FAR1e-2 | TPR@FAR1e-3 | Pairs |
|---------|----------|-----|-------------|-------------|-------|
| lfw | 71.47% | 28.53% | 18.50% | 7.03% | 6000 |
| rmfrd | 52.48% | 47.52% | 2.73% | 1.24% | 806 |

![Verification](outputs/figures/full/verify_metrics.svg)

## Notes

- Paper trains on WebFace-MASK / MS1MV3 / Glint360k; we use CelebA + synthetic mask aug under the 5 GiB gate.
- RMFRD pairs are built from common AFDB face/masked identity folders (see `DEVIATIONS.md`).
