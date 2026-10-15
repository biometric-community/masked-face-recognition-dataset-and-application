# Masked Face Recognition Dataset and Application

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Wang et al. (TBIOM 2023): CosFace ResNet-18 with mask augmentation on CelebA, evaluated on LFW and RMFRD masked↔unmasked pairs.

**Repository:** https://github.com/biometric-community/masked-face-recognition-dataset-and-application

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/smoke.sh
bash scripts/train_full.sh
```

Data roots (monorepo): `projects/datasets/celeba`, `lfw`, `rmfrd`.

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
