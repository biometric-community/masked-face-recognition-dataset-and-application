# IMPLEMENTATION_PLAN — Masked Face Recognition

## Goal
Reproduce the paper’s CosFace/ResNet masked-FR benchmark story under the monorepo size gate.

## Figure inventory
| Paper fig / table | Type | Our artifact |
|-------------------|------|--------------|
| Table IV (RMFRD / LFW) | verification Acc / TPR | `outputs/figures/*/verify_metrics.svg` |
| Training curves (implied) | loss vs epoch | `train_loss.svg` |

## Pipeline
1. Train CosFace ResNet-18 on CelebA with mask_aug_prob.
2. Eval LFW pairs + RMFRD masked↔unmasked pairs.
3. Report + fidelity Pass 1–5.

## Size gate
celeba + lfw + rmfrd/labeled ≈ 4.2 GiB → full_train.
