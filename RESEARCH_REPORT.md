# Benchmarking Semantic Segmentation for Liver & Tumor CT (LiTS) — Research Report

**Course/context:** CSE 445 Computer Vision — Assignment Part 1 (also serving as a research project)
**Task:** 2D semantic segmentation of liver and tumor on the LiTS 256×256 dataset, benchmarking three
architecturally distinct models under an identical, leakage-safe protocol.
**Status:** methodology fixed; medical-grade recipe applied; results tables have placeholders for the
final (v2) runs.

---

## 1. Clinical motivation
Automatic liver-lesion segmentation supports tumor-burden quantification and treatment planning. The
safety-critical failure is a **missed lesion (false negative)** — so this project prioritizes **tumor
sensitivity**, not just aggregate overlap. The gold standard for this data is 3D nnU-Net on raw
Hounsfield-Unit volumes; the assignment constrains us to **2D** and **three specific models**, and the
public Kaggle dataset is already flattened to **8-bit windowed PNG slices** (HU + 3D context lost before
we start). This report documents the **best medically-principled pipeline achievable within that box.**

## 2. Dataset & EDA (Notebook 0)
- LiTS 256×256: **131 patient volumes**, **58,638** axial slices; images + two **binary** masks
  (`Liver_mask`, `Tumor_mask`, values 0/255) fused into a class-index label `bg=0, liver=1, tumor=2`
  (tumor precedence).
- **Pixel imbalance (extreme):** background 97.72% / liver 2.16% / tumor **0.12%** (~760:1 bg:tumor).
- **Slice prevalence:** 32.7% contain liver/tumor; **67.3% are pure background** → filtered to the
  **19,158 liver-containing slices** for training/eval (rebalances to ~93/6.6/0.4%).
- **Near-duplicate analysis:** adjacent same-volume slices have mean perceptual-hash Hamming **1.9**
  (95% near-duplicate) vs **14.2** cross-volume — the empirical basis for volume-grouped splitting.

## 3. Methodology
### 3.1 Leakage-safe split (the critical design choice)
Split **by patient/volume** (not by slice), 70/15/15 → **92/20/19 volumes** (13,447 / 2,553 / 3,158
slices), fixed `SEED=42`, asserted zero volume overlap, saved once to `split.json` and **reused
identically** by all three models. This prevents near-identical adjacent slices of one patient from
leaking across splits — the #1 methodological error in medical ML.

### 3.2 Label fusion & filtering
Two binary masks → single 3-class map, tumor written last. Slices filtered to those containing
liver/tumor (per-slice, applied within each split → remains leakage-safe).

### 3.3 Medical-grade training recipe (identical across NB1/NB2; YOLO uses its native trainer)
- **Loss = Focal Tversky (α=0.7 > β=0.3, γ=1.33) + class-weighted CrossEntropy.** Tversky's asymmetry
  penalizes **false negatives** (missed tumor) harder than false positives → **sensitivity-first**.
  Class weights = median-frequency balancing on the **train split only**.
- **Augmentation (Albumentations, train-only, synchronized img+mask): NO flips.** The liver is
  *lateralized* (patient's right) and axial CT has fixed orientation, so H/V flips are anatomically
  implausible (and empirically caused left/right confusion in the baseline). Kept: mild affine
  (±15° rot, scale, shift), elastic/grid distortion (soft-tissue), brightness/contrast + Gaussian noise
  (CT windowing / scanner robustness). YOLO trained with `fliplr=flipud=0`.
- **Transfer learning:** DeepLabV3 (COCO), SegFormer-B0 (ADE20K), YOLOv26-sem (Cityscapes) pretrained
  backbones, fine-tuned end-to-end for 3 classes. AdamW + linear-warmup→cosine, ≥50 epochs, best
  checkpoint by validation mIoU (guards against the late-epoch over-confidence seen in the curves).

### 3.4 Models
| Model | Family | Params | Source |
|---|---|---|---|
| DeepLabV3-ResNet50 | CNN + ASPP | 42 M | torchvision |
| SegFormer-B0 | Hierarchical transformer (MiT-B0) + All-MLP head | 3.7 M | HuggingFace |
| YOLOv26-sem | Real-time detector adapted to per-pixel | ~small | Ultralytics |

## 4. Evaluation protocol
- **Required (Task F):** mIoU, per-class IoU, overall & mean pixel accuracy, Dice, pixel confusion matrix.
- **Clinical (added):** **tumor sensitivity/recall** (miss rate) and **per-patient Dice (mean ± std)**
  over test volumes — exposes per-patient variance that pooled pixel metrics hide.
- Test set touched once; all three models evaluated on the identical held-out volumes.

## 5. Ablation study: baseline vs medical-grade
Controlled: same split, models, epochs; single change = {loss, flips}. Baseline = first committed runs.

| Model | Recipe | mIoU | tumor IoU | tumor Dice | tumor sensitivity |
|---|---|---|---|---|---|
| DeepLabV3 | Baseline (Dice+CE, flips) | 0.751 | 0.376 | 0.546 | — (not measured in v1) |
| DeepLabV3 | **Medical-grade (Focal Tversky, no flip)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| SegFormer-B0 | Baseline | 0.761 | 0.411 | 0.583 | — |
| SegFormer-B0 | **Medical-grade** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| YOLOv26-sem | Medical-grade | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Hypothesis:** the medical-grade recipe raises **tumor sensitivity** (Focal Tversky) and reduces
left/right confusion (no flips), possibly trading a little precision — the expected clinical trade-off.

## 6. Error analysis (baseline observations, to re-confirm on v2)
- Dominant error: **tumor→liver** (50–54% of tumor pixels) — lesions sit inside the liver, are small,
  share intensity. Confusion matrices confirm this across models.
- **Distinct failure modes:** DeepLabV3 fails on tiny-liver edge slices (volume 106); SegFormer on
  left/right confusion (volume 20) — evidence that the two architectures fail *differently*.

## 7. Limitations
- **2D only** — no volumetric context; the clinical SOTA (3D nnU-Net) is out of scope by assignment rule.
- **No Hounsfield Units** — data is pre-windowed 8-bit PNG, so tissue-specific windowing can't be tuned.
- Tumor remains hard (~0.4% of pixels); small-lesion recall is the ceiling.

## 8. Future work (Part 2 hooks)
1. **2.5D input** — feed adjacent slices as channels to reclaim volumetric context within 2D models.
2. **Liver→tumor cascade** — segment liver first, restrict tumor search to the liver ROI (classic LiTS).
3. **3D nnU-Net** on the raw NIfTI volumes (if obtainable) as the true upper bound.
4. **Uncertainty / calibration** and **test-time augmentation** for deployment-grade confidence.
5. **Boundary-aware / compound losses** and lesion-level (not just pixel-level) detection metrics.

## 9. Reproducibility
- `SEED=42` (python/numpy/torch); split saved to `split.json` and reused; version-pinned imports.
- Hardware: Kaggle T4×2 GPU. Software: torch 2.10, transformers 5.0, albumentations 2.0.8, ultralytics.
- Artifacts per run: best checkpoint, `results.json` (all metrics), curves, confusion matrix, error grids.

## 10. Deliverables
Four public Kaggle notebooks — `eda-and-data-prep`, `seg-deeplabv3`, `seg-segformer-b0`,
`seg-yolov26-semantic` — each run end-to-end with outputs visible; NB3 contains the Task H comparison.

## References
Chen et al. (DeepLabV3, arXiv:1706.05587) · Xie et al. (SegFormer, NeurIPS 2021) · Ultralytics YOLO
semantic docs · Isensee et al. (nnU-Net) · Salehi et al. (Tversky loss) · Bilic et al. (LiTS Challenge).
