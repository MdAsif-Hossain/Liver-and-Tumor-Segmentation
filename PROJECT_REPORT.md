# LiTS Liver & Tumor Semantic Segmentation — Benchmark & Project Report

**Course:** CSE 348 Digital Image Processing — Assignment Part 1
**Department of Computer Science and Engineering, East West University**

**Group members**
1. Md. Asif Hossain — 2022-3-60-007
2. Nabil Subhan — 2022-3-60-063
3. Nudar — 2022-3-60-234

**Deliverables:** four Kaggle notebooks — `eda-and-data-prep`, `seg-deeplabv3`, `seg-segformer-b0`,
`seg-yolov26-semantic` — all run end-to-end with outputs. This document explains the whole pipeline and
answers the assignment's **Expected Coding Questions** against our actual code.

---

## 1. Executive summary

We benchmarked **three architecturally distinct semantic-segmentation models** — DeepLabV3-ResNet50 (CNN
+ ASPP), SegFormer-B0 (hierarchical transformer), and YOLOv26-sem (real-time detector adapted to per-pixel)
— on the **LiTS 256×256** liver+tumor CT dataset, under one **leakage-safe, identical** protocol.

**Final test results (identical Run-A recipe, held-out test = 19 patient volumes):**

| Model | mIoU | mean Dice | liver IoU | tumor IoU | tumor Dice | tumor sensitivity | params | train time |
|---|---|---|---|---|---|---|---|---|
| **DeepLabV3-R50** | **0.768** | **0.842** | 0.903 | **0.407** | **0.578** | **0.451** | 42 M | 256 min |
| SegFormer-B0 | 0.753 | 0.829 | 0.885 | 0.381 | 0.551 | 0.406 | **3.7 M** | **105 min** |
| YOLOv26-sem | 0.749 | 0.819 | **0.914** | 0.338 | 0.505 | 0.347 | 6.5 M | 356 min |

**Verdict:** **DeepLabV3 is the best** (highest mIoU and every tumor metric); **SegFormer-B0 is the most
efficient** (11× smaller, 2.4× faster, ~1.5 mIoU points behind); **YOLO** is best on liver but weakest on
tumor. Liver segmentation is strong (~0.90 IoU) for all; **tumor is the hard, imbalanced class** and the
main differentiator — the known ceiling of 2D LiTS.

---

## 2. Dataset & EDA (Notebook 0)

- **LiTS 256×256**, 3 folders: `Images/`, `Liver_mask/`, `Tumor_mask/`, **58,638 files each**, from
  **131 patient volumes** (`volume-{V}_{S}.png`).
- **Masks are two separate binary (0/255) PNGs** (liver, tumor). We **fuse** them into one class-index
  label: `label=0; label[liver>0]=1; label[tumor>0]=2` (**tumor written last → precedence**).
- **Pixel imbalance (extreme):** background **97.7%** / liver **2.2%** / tumor **0.13%** (~760:1 bg:tumor).
- **67.3% of slices are pure background** → we filter to the **19,158 liver/tumor-containing slices**.
- **Near-duplicate check:** adjacent same-volume slices have mean perceptual-hash Hamming **1.9** (95.4%
  near-duplicate) vs **14.2** cross-volume — the empirical justification for volume-grouped splitting.
- **Split (Task B):** by **volume/patient**, 70/15/15 → **92/20/19 volumes** = 13,447 / 2,553 / 3,158
  slices, seed 42, `LEAKAGE CHECK PASSED` (asserted zero volume overlap), saved to `split.json`.

## 3. Method (what the notebooks do)

- **Shared data pipeline (defined in NB0, reused in NB1–NB3):** fuse masks → `{0,1,2}`; **2.5D input**
  (3 channels = adjacent slices `[i−1, i, i+1]`, real volumetric context); Albumentations augmentation
  (train-only); **tumor-slice oversampling** (WeightedRandomSampler, 4×).
- **Loss:** `Dice + light class-weighted CrossEntropy` (weights `[0.3, 1, 6]`); DeepLabV3 adds `0.4×`
  auxiliary-head loss.
- **Optimisation:** AdamW, warmup(3)→cosine, weight decay **1e-2** (unified across NB1/NB2), AMP, ≥50 epochs.
- **Model selection:** best checkpoint by **validation tumor-F2** (recall-weighted), not mIoU.
- **Inference:** horizontal-flip **TTA**.
- **Metrics (Task F):** one pixel confusion matrix → mIoU, per-class IoU, pixel accuracy, mean pixel
  accuracy, Dice; plus **tumor sensitivity** and **per-patient Dice** (clinical).
- **YOLO (NB3):** the split is exported to YOLO's semantic format (`images/{train,val,test}` +
  `masks/{...}`, `data.yaml`), trained with the native Ultralytics `-sem` trainer, then evaluated with the
  *same* confusion-matrix code as the other two for a fair comparison.

## 4. Results, figures & ablation

**Task H figures (NB3):** summary table, overall+clinical bar chart, per-class IoU bar, radar chart,
side-by-side qualitative grid (all 3 models + GT), cross-model failure scatter, side-by-side confusion
matrices, and a per-patient tumor-Dice box plot.

**Cross-model failure analysis:** per-image IoU across models correlates at **r ≈ 0.91–0.93** (models
agree on which slices are hard → *data-driven* difficulty), yet only **5 of each model's worst-30 slices
are shared by all three** (worst-case tail is partly *model-specific*).

**Ablation study (controlled: same split/models/epochs, one recipe change):**

| Model | recipe | mIoU | tumor Dice | tumor sensitivity |
|---|---|---|---|---|
| DeepLabV3 | v1 baseline (flips, Dice+CE, mIoU-checkpoint) | 0.751 | 0.546 | — |
| DeepLabV3 | v2 (no-flip, Focal-Tversky) | 0.732 | 0.489 | 0.34 |
| DeepLabV3 | **v3 Run-A (2.5D + oversample + F2 + flips)** | **0.768** | **0.578** | **0.451** |

v2 **regressed** (removing flips → overfitting; the Focal-Tversky was mis-parameterised); v3 fixes the
regularisation, adds 2.5D context and recall-aligned selection, and beats the baseline.

---

## 5. Answers to the Expected Coding Questions (PDF §6.1)

### 5.1 Data Pipeline (Notebook 0)

**Q: How did you load and parse the mask format for your specific dataset?**
The dataset gives **two separate binary PNGs** per slice: `Liver_mask/livermask-{V}_{S}.png` and
`Tumor_mask/tumormask-{V}_{S}.png`, each with pixel values `{0, 255}` (not a 3-class map). We read each with
PIL, threshold `>0` to boolean, and **fuse** into a single class-index label:
`lab = zeros(256,256); lab[liver>0]=1; lab[tumor>0]=2`. Tumor is written **after** liver, so lesion pixels
that also fall inside the liver mask get class 2 (tumor precedence). We pair the three folders by the
integer tuple `(volume, slice)` parsed from the filename.

**Q: Walk through your split code; how did you verify no image is in more than one split; what seed?**
We split **by patient volume**, not by slice, because adjacent slices of one patient are near-identical
(near-dup Hamming ≈ 1.9). Code: take the sorted unique volume IDs, `np.random.RandomState(42).shuffle`
them, take the first 70% as train volumes, next 15% val, last 15% test (→ 92/20/19 volumes). We then
assert `train_vols & val_vols`, `train_vols & test_vols`, `val_vols & test_vols` are all **empty**, and a
second slice-level assert (`vset(train) & vset(val) == ∅`, etc.) prints `LEAKAGE CHECK PASSED`. **Seed =
42** — a fixed constant so the split is fully reproducible and byte-identical across all three model
notebooks (saved once in `split.json` and reused).

**Q: Show and explain each Albumentations transform (geometry/photometry, why, probability).**
Train-only pipeline (`train_tf`):
- `HorizontalFlip(p=0.5)` / `VerticalFlip(p=0.5)` — geometric mirror; strong, cheap regularisation
  (our ablation showed flips are the dominant regulariser here).
- `Affine(scale 0.9–1.1, translate 6%, rotate ±15°, p=0.5)` — patient size/position & gantry variation;
  small rotations keep anatomy plausible (mask uses nearest-neighbour).
- `OneOf(ElasticTransform(α=40,σ=6), GridDistortion, p=0.25)` — soft-tissue deformation, realistic for
  deformable abdominal organs, topology-preserving.
- `RandomBrightnessContrast(0.2, 0.2, p=0.3)` — **photometric, image-only**; mimics CT windowing/exposure.
- `GaussNoise(p=0.2)` — scanner-noise robustness, image-only.
- `Normalize(ImageNet)` + `ToTensorV2()`. Validation/test use **only** Normalize+ToTensorV2 (no aug).

**Q: How does `__getitem__` ensure image and mask get the same random transform each call?**
We call **one** `A.Compose(...)` with **both** arguments in a single call: `out = self.tf(image=img,
mask=lab)`. Albumentations samples the random parameters **once per call** and applies the *same*
geometric transform to `image` and `mask` together (masks are resampled with nearest-neighbour, and
photometric transforms are automatically skipped on the mask). This is what guarantees pixel-perfect
alignment — there is no separate transform for the mask.

**Q: What does your sanity-check visualization confirm, and what would a bug look like?**
Task D shows post-augmentation training samples as `image | mask | overlay` (center slice of the 2.5D
input). It confirms (1) image↔mask alignment after flips/affine/elastic (overlay lands on the correct
anatomy), (2) consistent colour mapping (liver=gold, tumor=red), (3) augmentations don't destroy mask
regions, and (4) labels are `{0,1,2}`. A **bug** would look like: mask shifted/mirrored relative to the
image (async transform), tumor appearing outside the liver, wrong class colours, or empty masks.

### 5.2 DeepLabV3 (Notebook 1)

**Q: Which torchvision constructor and why? Backbone trade-offs?**
`torchvision.models.segmentation.deeplabv3_resnet50(weights=DEFAULT, aux_loss=True)`. We chose **ResNet-50**
as the balanced default: **MobileNetV3 (~11 M)** is fastest but least accurate; **ResNet-50 (~42 M)** is
the accuracy/compute sweet spot for a Kaggle T4; **ResNet-101 (~61 M)** is more accurate but slower and
heavier. ResNet-50 fit our GPU/time budget while giving strong capacity.

**Q: How did you adapt the pretrained classifier head for your class count?**
The COCO-pretrained model predicts 21 classes. We replace the final `1×1` convs:
`model.classifier[-1] = nn.Conv2d(256, 3, 1)` and `model.aux_classifier[-1] = nn.Conv2d(256, 3, 1)`. The
backbone + ASPP weights transfer; only these two projection layers are new (randomly initialised) and
learn the 3 LiTS classes during fine-tuning.

**Q: What is the ASPP module and why does it help?**
**Atrous Spatial Pyramid Pooling** applies several parallel atrous (dilated) convolutions at different
dilation rates plus a global image-pooling branch, then concatenates them. This samples the feature map at
**multiple receptive-field scales simultaneously** without downsampling, so the head sees both small
lesions and large liver context — ideal for LiTS where liver and tumor vary enormously in size.

**Q: What loss? If class-weighted, how were weights computed from the EDA histogram?**
**Dice + class-weighted CrossEntropy** (`ce(out,y) + dice(out,y)`, plus `0.4×` the same on the aux head).
From the EDA pixel counts we computed **median-frequency** weights: `freq_c = pixels_c / total`,
`w_c = median(freq)/freq_c` → `[bg 0.07, liver 1.0, tumor 16.5]`. We deliberately **tempered** these to a
lighter `[0.3, 1.0, 6.0]` in the final recipe because the raw 16.5 tumor weight caused high-variance
gradient spikes and amplified overfitting on the rare class; we instead push tumor recall via
**oversampling** and **F2-based checkpoint selection**. Dice is inherently imbalance-robust.

**Q: What optimizer / LR schedule? What does the schedule do at each step?**
**AdamW** (lr 1e-4, weight-decay 1e-2). Schedule = `SequentialLR([LinearLR warmup for 3 epochs] →
[CosineAnnealingLR for the rest])`, stepped **once per epoch**. During the first 3 epochs the LR ramps
linearly from 10%→100% of base (stabilises the freshly-initialised head); afterwards it follows a cosine
curve decaying smoothly toward ~0 by the final epoch (lets the model settle into a sharp minimum).

**Q: Walk through the training loop: `model.train()` vs `model.eval()` and why it matters.**
Each epoch: `model.train()` (enables dropout and lets BatchNorm update its running statistics), iterate
the train loader with AMP autocast + GradScaler, `criterion` on `out["out"]` + `0.4×` on `out["aux"]`,
backward, `scaler.step`. Then `run_eval` calls **`model.eval()`** (freezes BN running stats and disables
dropout) inside `torch.no_grad()` so validation is deterministic and uses population statistics — using
`train()` at eval would leak batch statistics and give unstable, optimistic numbers. We save the
checkpoint at the best **val tumor-F2**.

### 5.3 SegFormer-B0 (Notebook 2)

**Q: What pretrained checkpoint and why that domain?**
`nvidia/segformer-b0-finetuned-ade-512-512` — pretrained on **ADE20K** (150-class general scenes). We chose
the general-scene checkpoint over the Cityscapes (urban-driving) one because abdominal CT is generic
top-down imagery, not street data; ADE20K's diverse features transfer better than street-specific ones.

**Q: Explain `ignore_mismatched_sizes=True` — which layers are re-initialized and why?**
ADE20K has 150 classes so the checkpoint's `decode_head.classifier` is `Conv2d(256, 150, 1)`; we need 3
classes (`Conv2d(256, 3, 1)`). Loading normally would error on the shape mismatch. `ignore_mismatched_sizes
=True` tells HuggingFace to **skip and randomly re-initialise only the mismatched layers** — here the
decode-head `classifier` weight **and** bias — while loading the entire MiT-B0 encoder and MLP-fusion
weights intact. The run log confirms exactly these two tensors are re-initialised.

**Q: What is the MiT-B0 encoder and how does it differ from a standard ViT?**
The **Mix-Transformer (MiT)** is a hierarchical vision transformer with three key differences from a plain
ViT: (1) **overlapping** patch embeddings (convolutional, preserving local continuity) instead of ViT's
non-overlapping patches; (2) a **4-stage hierarchy** producing multi-scale feature maps (1/4…1/32) rather
than ViT's single-scale tokens; (3) **efficient self-attention** with spatial reduction, cutting the
quadratic cost. It also needs no positional embeddings. This makes it well-suited to dense prediction.

**Q: What is the All-MLP decoder and how does it fuse multi-stage features?**
SegFormer's decoder is intentionally lightweight: each of the 4 encoder stages is passed through a small
**MLP** to a common channel width, **bilinearly upsampled** to 1/4 resolution, **concatenated**, and fused
by another MLP that predicts per-pixel logits — no heavy convolutional decoder. Because the MiT already
provides multi-scale features, a simple MLP fusion suffices and is fast.

**Q: HuggingFace loss vs your own loss — which and why?**
We compute **our own** loss. We call the model as `model(pixel_values=x).logits` **without** passing
`labels=`, so HF does not compute its internal cross-entropy. We take the `H/4×W/4` logits, bilinearly
upsample to 256², and apply the **identical `Dice + weighted-CE`** used in NB1. Reason: a fair benchmark
requires the **exact same loss** across models — using HF's built-in loss would introduce a confound.

**Q: How were images resized/padded, and where?**
No resizing is needed — the dataset is natively **256×256** and the dataloader emits 256×256 tensors.
SegFormer internally outputs logits at **64×64** (stride 4); we upsample those back to 256×256 with
`F.interpolate(..., mode="bilinear")` inside our `forward_logits` wrapper (i.e., in our forward code, not
inside the HF model and not in the dataloader), so loss and metrics are computed at full mask resolution.

### 5.4 YOLOv26 Semantic (Notebook 3)

**Q: Difference between `-sem` and `-seg`; why does the wrong one change the task?**
`-sem` = **semantic segmentation** (dense per-pixel class map — what we need). `-seg` = **instance
segmentation** (per-object masks + bounding boxes + confidences). Ultralytics infers the task from the
checkpoint name, so a `-seg` checkpoint would train an instance model producing boxes/instance masks and
instance metrics (mask mAP), not a per-pixel semantic map — a fundamentally different task, output, and
metric. We used `yolo26s-sem.pt`.

**Q: What does your `data.yaml` contain? Walk through each field.**
```yaml
path: /kaggle/working/yolo_ds   # dataset root
train: images/train             # train images (relative to path)
val:   images/val               # val images
test:  images/test              # held-out test images
masks_dir: masks                # folder holding the per-image masks (same stem)
names: {0: background, 1: liver, 2: tumor}   # class-id → name
```
Masks live in `masks/{train,val,test}/` with the **same filename stem** as each image.

**Q: What mask format does YOLOv26-sem expect and how did you verify it?**
Single-channel PNG where **pixel value = class ID** (0/1/2) and **255 = ignore**. We generate masks from
`combined_label()` (values strictly `{0,1,2}`) and save with `Image.fromarray(uint8)` → single-channel L
PNGs. We use **no** 255/ignore pixels (every pixel is labelled). Verification: the fused labels only ever
contain `{0,1,2}` (checked in NB0's sanity grid), and YOLO's own validation reports the three named
classes correctly.

**Q: What does `imgsz` control and how did you pick it?**
`imgsz` is the square resolution YOLO resizes inputs to for training/inference. We set **`imgsz=256`** to
match the dataset's **native** 256×256 — this avoids any up/down-scaling that would blur the already-tiny
tumors. (YOLO's Cityscapes default is 1024; we deliberately did not upscale to keep the 2.5D depth cue and
save compute.)

**Q: How do you read back `result.semantic_mask.data` and convert it to mIoU?**
For each test image we run `model.predict(path, imgsz=256)[0]`, read `result.semantic_mask.data` → an
`(H,W)` tensor of integer class IDs, convert to NumPy, resize to 256² with nearest-neighbour if needed,
and **accumulate it into a 3×3 confusion matrix** against the ground-truth fused mask (via
`np.bincount(3*gt + pred)`). mIoU is then `mean(TP_c / (TP_c+FP_c+FN_c))` over the 3 classes — the same
`ConfMat` code used for DeepLabV3 and SegFormer, so all three are measured identically.

**Q: What optimizer/LR schedule does Ultralytics use by default, and did you override any?**
Ultralytics default for the `-sem` task: `optimizer=auto` (it selects SGD or AdamW based on the run),
`lr0=0.01` with a **cosine** schedule (`lrf=0.01`), `momentum=0.937`, `weight_decay=5e-4`, `warmup_epochs=3`.
We **kept** the optimizer/LR defaults (documented in the config cell) and only overrode **augmentation** to
match our recipe: `fliplr=0.5, flipud=0.5` (flips on) and `hsv_h=hsv_s=hsv_v=0` (colour jitter off, because
our 3 channels encode neighbouring slices/depth, not colour).

### 5.5 Error Analysis & Final Comparison (Task H)

**Q: Which model achieved the best mIoU and why for this data?**
**DeepLabV3 (0.768).** Two reasons: (1) our fairness fix — unifying weight-decay to 1e-2 — most benefited
the **large, previously under-regularised** 42 M CNN, and it had the capacity to exploit the 2.5D context
and oversampling; (2) **ASPP's multi-scale receptive field** matches LiTS's huge size variation between
liver and tiny tumors. The small 3.7 M SegFormer was already well-regularised, so the same changes helped
it less — which is why the ranking flipped from earlier baselines.

**Q: Which class was hardest across all three, and why?**
**Tumor** (IoU 0.34–0.41, sensitivity 0.35–0.45). Statistical reason: it is only **~0.4%** of pixels even
after filtering (extreme imbalance). Visual reasons: lesions are **small, low-contrast, and sit inside the
liver** with similar intensity, so all three models leak **tumor→liver** (48–60% of tumor pixels). Per-
patient tumor Dice also has huge variance (~0.0 to ~0.85), i.e. some patients' lesions are essentially
missed.

**Q: Do the three models fail on the same images or different ones? What does that tell you?**
**Mostly the same, with model-specific tails.** Per-image IoU correlates at **r ≈ 0.91–0.93** across every
pair → the models agree strongly on which slices are easy/hard, so difficulty is **data-driven** (tiny-
tumor / liver-edge slices are hard for everyone). But only **5 of each model's worst-30 slices are shared
by all three** (pairwise Jaccard 0.18–0.28) → the **extreme** failures are partly architecture-specific.
Interpretation: the data ceiling dominates, but architecture still shapes the worst-case behaviour.

**Q: To improve the worst model, what specific change and why?**
The weakest is **YOLOv26-sem** on tumor. Concrete changes: (1) **increase `imgsz` to 512** — YOLO's
detector-lineage downsampling destroys tiny lesions, so higher resolution preserves them; (2) use a
**larger `-sem` variant** (`yolo26m/l-sem`) for more capacity; (3) a **liver-ROI cascade** — segment the
liver first and restrict tumor search to that ROI, which removes most background/other-organ distractors;
(4) if the trainer allows, add a tumor-weighted region loss. The single highest-leverage change is (1)+(3):
resolution + ROI focus directly target the tiny-tumor failure mode.

---

## 6. Limitations & future work (Part 2 directions)

- **2D only.** The clinical gold standard for this data is **3D nnU-Net** on raw Hounsfield-Unit volumes;
  the assignment and the pre-windowed 8-bit PNGs constrain us to 2D. This is the main tumor-accuracy ceiling.
- **No HU values** (data is pre-windowed), so tissue-specific windowing can't be tuned.
- **Tumor generalisation gap:** validation tumor recall (~0.70) drops to test (~0.45) — inter-patient
  heterogeneity on a 19-volume test set.
- **Next steps:** liver→tumor cascade, higher-resolution / true 3D input, uncertainty estimation, and a
  precision–recall operating-point analysis for clinical deployment.

## 7. Reproducibility

`SEED=42` (python/numpy/torch); split saved once and reused; version-pinned imports; Kaggle T4×2 GPU;
torch 2.10, transformers 5.0, albumentations 2.0.8, ultralytics 8.4.87. Every run saves its best checkpoint,
`results.json`, curves, confusion matrix and error grids.

## Appendix — brief theory notes (viva support)

- **IoU** = |A∩B| / |A∪B|; **Dice** = 2|A∩B| / (|A|+|B|) = F1 at pixel level; mIoU is preferred over pixel
  accuracy on imbalanced data because background dominates accuracy but not IoU.
- **Focal loss** down-weights easy pixels to focus on hard/rare ones; class weights = `median(freq)/freq_c`.
- **Encoder–decoder:** encoder compresses to semantic features (loses spatial detail); decoder recovers
  resolution; **skip connections** (U-Net) re-inject high-resolution detail lost in downsampling.
- **Atrous/dilated convolution** enlarges the receptive field without extra parameters or downsampling;
  **output stride** is input/feature size ratio — lower stride = finer masks at higher compute.
- **Data leakage** here = same patient's near-identical slices split across train/test → we split by volume.
- **Transfer learning** (pretrained backbone) beats random init because low-level edge/texture features are
  universal, so the model converges faster and generalises better on a small dataset.
