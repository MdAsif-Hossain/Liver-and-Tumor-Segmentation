# Universal Run Guide — how to run every notebook, in order

This is the single source of truth for **what to attach, which GPU to pick, and what order to run** across all
three projects. Everything runs on **Kaggle** (the local repo is for authoring only). For each notebook: set it
**Public**, **Run All**, confirm outputs are visible, then **Save Version**.

Legend: 🖥️ = GPU setting · 📎 = datasets to attach · ▶️ = run order · 📤 = output to publish as a Kaggle Dataset.

---

## Project 1 — Assignment 1 (Part A): supervised benchmark  *(already complete)*

🖥️ **GPU: T4 ×2** (or P100) · Internet: **On** (downloads pretrained weights).

| # | Notebook | 📎 Attach | Notes |
|---|---|---|---|
| ▶️1 | `eda-and-data-prep` | LiTS 256×256 dataset | 📤 publish output → contains `split.json`, `class_weights.json`, `manifest.csv` |
| ▶️2 | `seg-deeplabv3` | LiTS + NB0 output | reads `split.json`; appends to `results.json` |
| ▶️3 | `seg-segformer-b0` | LiTS + NB0 output | same split |
| ▶️4 | `seg-yolov26-semantic` | LiTS + NB0 output | Task-H final comparison |

**LiTS dataset:** `sodko3/lits-dataset-liver-and-tumor-segmentation-256x256` (search Kaggle "LiTS 256x256").
**Critical:** NB1–4 must all attach the **same NB0 output** so the split is byte-identical.

---

## Project 2 — Assignment 2 (Part B): self-supervised  *(building now)*

🖥️ **GPU: P100** (single, stable) **or T4 ×2** (bigger batch) · Internet: **On** (ImageNet / DINOv2 checkpoints).
Each method notebook is standalone: SSL-pretrain (50 ep) → transfer encoder → fine-tune on the small **val** split
(50 ep) → evaluate on **test** split.

| # | Notebook | 📎 Attach | 🖥️ | Notes |
|---|---|---|---|---|
| ▶️1 | `lits-ssl-data-prep` | LiTS + **Part-A NB0 output** (`split.json`) | T4×2 | reuses Part-A split, remaps SSL roles. 📤 publish → `partB_split_roles.json` |
| ▶️2 | `lits-simclr` | LiTS + **Part-B data-prep output** | P100/T4×2 | ResNet-50, NT-Xent → DeepLabV3. **Run this first** — validates the whole pipeline; BYOL/MAE copy it |
| ▶️3 | `lits-byol` | LiTS + data-prep output | P100/T4×2 | ResNet-50 online+EMA target |
| ▶️4 | `lits-mae` | LiTS + data-prep output | P100/T4×2 | ResNet-50 CNN-MAE (no adapter) |
| ▶️5 | `lits-dinov2` | LiTS + data-prep output | P100/T4×2 | ViT-S/14 + token→grid adapter; runs the frozen-vs-fine-tuned experiment |
| ▶️6 | `lits-partB-final-comparison` | data-prep output + all method outputs + **Part-A `results.json`** | none/T4 | label-efficiency table + chart |

**If a ViT method's epoch >10 min:** split into `…-pretrain` (saves encoder as a 📤 dataset) + `…-downstream`
(attaches that encoder). Report the measured per-epoch time in the config cell either way.

**Run budget:** SSL 50 ep + fine-tune 50 ep per method. Run methods on **separate days** if you hit Kaggle's
weekly GPU quota. Each method appends to its own `results.json`; the final notebook merges them.

---

## Project 3 — Research: single-image dehazing (Dark Channel Prior)  *(building now)*

🖥️ **GPU: None needed** for DCP (pure CPU, seconds/image) — pick **GPU only** if you run the optional learned
reference (AOD-Net/FFA-Net). Internet: **On** (only for the optional learned baseline weights).

| # | Notebook | 📎 Attach | 🖥️ | Notes |
|---|---|---|---|---|
| ▶️1 | `dehaze-dcp-pipeline` | RESIDE-SOTS **or** O-HAZE (else built-in synthetic-haze fallback runs) | None | implements DCP; stage-by-stage figures; sanity PSNR/SSIM |
| ▶️2 | `dehaze-ablation-eval` | same dehazing dataset | None | the **research** notebook: full ablation sweep → tables + figures |
| ▶️3 | `dehaze-learned-reference` *(optional)* | same + pretrained dehaze weights | GPU | positions DCP vs a deep model |

**Datasets (search Kaggle):** "RESIDE SOTS" / "RESIDE standard" (synthetic, perfectly paired GT — **primary**),
"O-HAZE" and "I-HAZE" (real haze pairs), "Dense-Haze NTIRE 2019" (hard cases). If none is attached, both notebooks
**auto-fall back** to synthetic haze generated from clear images via the haze model `I = J·t + A(1−t)`, so they run
anywhere — but attach RESIDE for the reportable numbers.

---

## Global checklist before every submission
- [ ] Notebook **Public**, **Run All** with **all outputs visible**, then **Save Version**.
- [ ] Correct dataset(s) attached and the **exact path** resolves (each notebook prints what it found).
- [ ] GPU set as above (or None where noted) so you don't burn quota on CPU-only work.
- [ ] Outputs that feed a later notebook are **published as a Kaggle Dataset** and attached downstream.
