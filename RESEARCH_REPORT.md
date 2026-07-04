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

### 3.3 Training recipe (Run A — identical across NB1/NB2; YOLO mirrors it in its native trainer)
Derived from a multi-agent ("Claude council") review that found the first medical-grade recipe
**overfit** (removing flips gutted regularization) and its Focal Tversky was **mis-parameterized**
(inverted focal exponent, asymmetry washed out by a symmetric weighted-CE). The corrected recipe:
- **Input = 2.5D:** 3 channels = adjacent slices `[i−1, i, i+1]` (real volumetric context; replaces
  grayscale-replicated channels). Leakage-safe — neighbours are same-volume = same split.
- **Augmentation (Albumentations, train-only): flips RESTORED** — the dominant regularizer for this
  small dataset (ablation-confirmed) — + affine, elastic/grid distortion, brightness/contrast, noise.
- **Tumor-slice oversampling** (WeightedRandomSampler, tumor slices 4×) → per-batch tumor prevalence
  ~12% → ~45%.
- **Loss = Dice + lightly class-weighted CrossEntropy** (tumor weight 6, not extreme — avoids gradient
  spikes that amplify overfitting).
- **Checkpoint selection by validation tumor F2** (recall-weighted) — the clinical objective, instead of
  mIoU (which selects the least tumor-sensitive epoch).
- **Test-time augmentation** (horizontal flip) at inference.
- **Weight decay unified at 1e-2** across models (fixes an earlier fairness gap).
- **Transfer learning:** DeepLabV3 (COCO), SegFormer-B0 (ADE20K), YOLOv26-sem (Cityscapes), ≥50 epochs,
  AdamW + linear-warmup→cosine.

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
| DeepLabV3 | v1 Baseline (Dice+CE, flips) | 0.751 | 0.376 | 0.546 | — |
| DeepLabV3 | v2 (Focal Tversky, no flips) | 0.732 | 0.323 | 0.489 | 0.34 |
| DeepLabV3 | **v3 Run-A (2.5D + oversample + F2 + flips)** | **0.768** | **0.407** | **0.578** | **0.451** |
| SegFormer-B0 | v1 Baseline | 0.761 | 0.411 | 0.583 | — |
| SegFormer-B0 | v2 | 0.747 | 0.376 | 0.546 | 0.40 |
| SegFormer-B0 | **v3 Run-A** | 0.753 | 0.381 | 0.551 | 0.406 |
| YOLOv26-sem | **v3 Run-A** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Findings.** (1) **v2 regressed** on every metric — removing flips caused overfitting (train loss
0.16→0.07, val loss diverged) and the Focal Tversky was mis-parameterized (inverted focal exponent,
asymmetry washed out by a symmetric weighted-CE). (2) **v3 Run-A fixed the overfitting** (val loss stable
~0.30, no divergence) and made **DeepLabV3 the best model** (tumor Dice 0.546→0.578, tumor sensitivity
0.34→0.451, tumor→liver leakage 0.60→0.49). (3) The bundle helped **DeepLabV3 far more than SegFormer**,
because its dominant lever — the weight-decay fix (1e-4→1e-2) — applied only to the under-regularized CNN;
the already-regularized 3.7M transformer stayed ~neutral. The model **ranking flipped**: DeepLabV3 now
leads on tumor and mIoU. (4) **Remaining limitation:** a val→test tumor-recall gap (~0.70 → ~0.45) and
high per-patient variance (tumor Dice 0.47±0.30) reflect patient heterogeneity — the ceiling of a small
2D patient-level split; 3D/more data is the Part-2 direction. All three final-benchmark models use the
identical v3 recipe for fairness; v1/v2 are the controlled ablation.

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
