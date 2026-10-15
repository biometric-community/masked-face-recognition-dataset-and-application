"""LFW and RMFRD verification pair builders."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _resolve(cfg: dict, key: str, default: str) -> Path:
    root = Path(cfg["data"].get(key, default))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    return root


def resolve_lfw(cfg: dict) -> Path:
    root = _resolve(cfg, "lfw_root", "../../datasets/lfw")
    if not (root / "lfw").is_dir() or not (root / "pairs.txt").is_file():
        raise FileNotFoundError(root)
    return root


def resolve_rmfrd(cfg: dict) -> Path:
    root = _resolve(cfg, "rmfrd_root", "../../datasets/rmfrd/AFDB")
    face = root / "AFDB_face_dataset"
    mask = root / "AFDB_masked_face_dataset"
    if not face.is_dir() or not mask.is_dir():
        raise FileNotFoundError(root)
    return root


def _tf(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _subsample_pairs(
    pairs: List[Tuple[str, str, int]], max_pairs: Optional[int]
) -> List[Tuple[str, str, int]]:
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    n_each = max_pairs // 2
    out = pos[:n_each] + neg[:n_each]
    if len(out) < max_pairs:
        rest = [p for p in pairs if p not in out]
        out.extend(rest[: max_pairs - len(out)])
    return out[:max_pairs]


def _parse_lfw(pairs_path: Path, max_pairs: Optional[int] = None):
    out = []
    for line in pairs_path.read_text().strip().splitlines()[1:]:
        toks = line.split()
        if len(toks) == 3:
            name, n1, n2 = toks
            a = f"{name}/{name}_{int(n1):04d}.jpg"
            b = f"{name}/{name}_{int(n2):04d}.jpg"
            out.append((a, b, 1))
        elif len(toks) == 4:
            n1, i1, n2, i2 = toks
            a = f"{n1}/{n1}_{int(i1):04d}.jpg"
            b = f"{n2}/{n2}_{int(i2):04d}.jpg"
            out.append((a, b, 0))
    return _subsample_pairs(out, max_pairs)


class PairDataset(Dataset):
    def __init__(self, root: Path, pairs, image_size: int = 112, img_subdir: str = ""):
        self.root = Path(root)
        self.base = self.root / img_subdir if img_subdir else self.root
        self.pairs = pairs
        self.transform = _tf(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        ta = self.transform(Image.open(self.base / a).convert("RGB"))
        tb = self.transform(Image.open(self.base / b).convert("RGB"))
        return ta, tb, int(y)


def build_lfw(cfg: dict, image_size: int, max_pairs: Optional[int] = None) -> PairDataset:
    root = resolve_lfw(cfg)
    pairs = _parse_lfw(root / "pairs.txt", max_pairs=max_pairs)
    return PairDataset(root, pairs, image_size=image_size, img_subdir="lfw")


def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def build_rmfrd_pairs(
    cfg: dict,
    image_size: int,
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> PairDataset:
    """Masked–unmasked verification pairs on common RMFRD identities.

    Positive: same ID, one face from AFDB_face_dataset and one from AFDB_masked_face_dataset.
    Negative: different IDs, masked vs unmasked.
    """
    root = resolve_rmfrd(cfg)
    face_root = root / "AFDB_face_dataset"
    mask_root = root / "AFDB_masked_face_dataset"
    face_ids = {p.name for p in face_root.iterdir() if p.is_dir()}
    mask_ids = {p.name for p in mask_root.iterdir() if p.is_dir()}
    common = sorted(face_ids & mask_ids)
    if not common:
        raise RuntimeError(f"No common IDs under {root}")

    rng = np.random.RandomState(seed)
    face_imgs = {i: _list_images(face_root / i) for i in common}
    mask_imgs = {i: _list_images(mask_root / i) for i in common}
    common = [i for i in common if face_imgs[i] and mask_imgs[i]]

    pairs: List[Tuple[str, str, int]] = []
    # positives
    for i in common:
        fa = face_imgs[i][rng.randint(0, len(face_imgs[i]))]
        mb = mask_imgs[i][rng.randint(0, len(mask_imgs[i]))]
        pairs.append((str(fa.resolve()), str(mb.resolve()), 1))

    # negatives
    for i in common:
        j = common[rng.randint(0, len(common))]
        while j == i:
            j = common[rng.randint(0, len(common))]
        fa = face_imgs[i][rng.randint(0, len(face_imgs[i]))]
        mb = mask_imgs[j][rng.randint(0, len(mask_imgs[j]))]
        pairs.append((str(fa.resolve()), str(mb.resolve()), 0))

    pairs = _subsample_pairs(pairs, max_pairs)
    # PairDataset with absolute paths: use empty root and full paths stored
    return AbsolutePairDataset(pairs, image_size=image_size)


class AbsolutePairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]], image_size: int = 112):
        self.pairs = pairs
        self.transform = _tf(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        ta = self.transform(Image.open(a).convert("RGB"))
        tb = self.transform(Image.open(b).convert("RGB"))
        return ta, tb, int(y)
