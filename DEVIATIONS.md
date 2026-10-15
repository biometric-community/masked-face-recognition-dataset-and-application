# DEVIATIONS — Masked Face Recognition Dataset and Application

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train data | WebFace-MASK / MS1MV3 / Glint360k | CelebA + synthetic lower-face mask aug | Size gate (&lt;5 GiB); large corpora unavailable |
| D2 | Backbone | R18–R100 CosFace/ArcFace | ResNet-18 CosFace | Local compute / smoke–full parity |
| D3 | Mask aug | Texture templates (Fig. 3) | Solid lower-face rectangle | No author mask assets in-repo |
| D4 | RMFRD protocol | Official 7178 verification pairs | Built masked↔unmasked pairs on 442 common AFDB IDs | Official pair archive not publicly downloadable here |
| D5 | Extra benches | AgeDB / CFP / ICCV2021-MFR | LFW + RMFRD only | AgeDB/CFP stubs; MFR not mirrored |
