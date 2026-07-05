# Expected Coding Questions — Answered

**Course:** CSE 348 Digital Image Processing · Assignment Part 1 — Viva preparation
**Group:** Md. Asif Hossain (2022-3-60-007) · Nabil Subhan (2022-3-60-063) · Nudar (2022-3-60-234)

Every question from the assignment's **§6.1 Expected Coding Questions** is answered below **against our
actual implementation**. See **[README.md](README.md)** for the results, figures and full method.

---

## 1. Data Pipeline (Notebook 0 — `eda-and-data-prep`)

**Q: How did you load and parse the mask format for your specific dataset?**
The dataset provides **two separate binary PNGs** per slice: `Liver_mask/livermask-{V}_{S}.png` and
`Tumor_mask/tumormask-{V}_{S}.png`, each with pixel values `{0, 255}` (not a 3-class map). We read each with
PIL, threshold `>0`, and **fuse** into a single class-index label:
`lab = zeros(256,256); lab[liver>0]=1; lab[tumor>0]=2`. Tumor is written **after** liver, so lesion pixels
inside the liver get class 2 (**tumor precedence**). The three folders are paired by the integer tuple
`(volume, slice)` parsed from the filenames.

**Q: Walk through your split code; how did you verify no image is in more than one split; what seed?**
We split **by patient volume**, not by slice (adjacent slices of one patient are near-identical, Hamming
≈ 1.9). Take the sorted unique volume IDs, `np.random.RandomState(42).shuffle` them, take the first 70% as
train volumes, next 15% val, last 15% test → 92/20/19 volumes. We then **assert** `train∩val`,
`train∩test`, `val∩test` are all empty (both at volume level and again at slice level), printing
`LEAKAGE CHECK PASSED`. **Seed = 42** — a fixed constant so the split is fully reproducible and byte-
identical across all three model notebooks (saved once to `split.json` and reused).

**Q: Show and explain each Albumentations transform (geometry/photometry, why, probability).**
Train-only `train_tf`:
- `HorizontalFlip(0.5)` / `VerticalFlip(0.5)` — geometric mirror; strong cheap regularisation (our ablation
  showed flips are the dominant regulariser here).
- `Affine(scale 0.9–1.1, translate 6%, rotate ±15°, p=0.5)` — patient size/position & gantry variation
  (mask resampled nearest-neighbour).
- `OneOf(ElasticTransform(α=40,σ=6), GridDistortion, p=0.25)` — realistic soft-tissue deformation,
  topology-preserving.
- `RandomBrightnessContrast(0.2, 0.2, p=0.3)` — **photometric, image-only**; mimics CT windowing/exposure.
- `GaussNoise(p=0.2)` — scanner-noise robustness, image-only.
- `Normalize(ImageNet)` + `ToTensorV2()`. Val/test use **only** Normalize + ToTensorV2 (no augmentation).

**Q: How does `__getitem__` ensure image and mask get the same random transform each call?**
We pass **both** to a **single** `A.Compose` call: `out = self.tf(image=img, mask=lab)`. Albumentations
samples the random parameters **once per call** and applies the same geometric transform to image and mask
together (masks via nearest-neighbour; photometric ops auto-skipped on the mask). This guarantees
pixel-perfect alignment — there is no separate mask transform to desynchronise.

**Q: What does your sanity-check visualization confirm, and what would a bug look like?**
Task D shows `image | mask | overlay` (center slice of the 2.5D input) post-augmentation. It confirms
(1) image↔mask alignment after flips/affine/elastic, (2) consistent colour mapping (liver=gold, tumor=red),
(3) augmentations don't destroy mask regions, (4) labels are `{0,1,2}`. A **bug** would appear as a mask
shifted/mirrored relative to the image, tumor outside the liver, wrong colours, or empty masks.

## 2. DeepLabV3 (Notebook 1 — `seg-deeplabv3`)

**Q: Which torchvision constructor and why? Backbone trade-offs?**
`deeplabv3_resnet50(weights=DEFAULT, aux_loss=True)`. **ResNet-50** is the balanced default: **MobileNetV3
(~11 M)** fastest/least accurate, **ResNet-50 (~42 M)** accuracy/compute sweet spot for a T4, **ResNet-101
(~61 M)** more accurate but slower. ResNet-50 fit our GPU/time budget with strong capacity.

**Q: How did you adapt the pretrained classifier head for your class count?**
COCO gives 21 classes; we replace the final 1×1 convs:
`model.classifier[-1] = nn.Conv2d(256, 3, 1)` and `model.aux_classifier[-1] = nn.Conv2d(256, 3, 1)`. The
backbone + ASPP transfer; only these two projections are new and learn the 3 LiTS classes.

**Q: What is the ASPP module and why does it help?**
**Atrous Spatial Pyramid Pooling** runs several parallel dilated convolutions at different rates plus a
global image-pooling branch and concatenates them — sampling the feature map at **multiple receptive-field
scales at once** without downsampling. That multi-scale context suits LiTS's huge size gap between the liver
and tiny tumors.

**Q: What loss? If class-weighted, how computed from the EDA histogram?**
**Dice + class-weighted CrossEntropy** (+ 0.4× the same on the aux head). From EDA pixel counts we computed
**median-frequency** weights `w_c = median(freq)/freq_c` → `[bg 0.07, liver 1.0, tumor 16.5]`, but
deliberately **tempered** them to `[0.3, 1.0, 6.0]` (the raw 16.5 caused high-variance gradient spikes /
overfitting on the rare class); tumor recall is instead pushed via **oversampling** + **F2 checkpointing**.

**Q: What optimizer / LR schedule? What does the schedule do at each step?**
**AdamW** (lr 1e-4, wd 1e-2). `SequentialLR([LinearLR warmup 3 epochs] → [CosineAnnealingLR])`, stepped per
epoch: first 3 epochs ramp LR 10%→100% (stabilise the new head), then a cosine decay toward ~0 by the final
epoch (settle into a sharp minimum).

**Q: Walk through the training loop: `train()` vs `eval()` and why it matters.**
Each epoch: `model.train()` (dropout on, BatchNorm updates running stats) + AMP + GradScaler,
`criterion(out["out"], y) + 0.4·criterion(out["aux"], y)`, backward, step. Validation uses **`model.eval()`**
inside `torch.no_grad()` (BN frozen to population stats, dropout off) → deterministic metrics; using
`train()` at eval would leak batch statistics and give unstable numbers. We save the best-val-tumor-F2
checkpoint.

## 3. SegFormer-B0 (Notebook 2 — `seg-segformer-b0`)

**Q: What pretrained checkpoint and why that domain?**
`nvidia/segformer-b0-finetuned-ade-512-512` — **ADE20K** (150-class general scenes). Chosen over the
Cityscapes (urban-driving) checkpoint because abdominal CT is generic imagery, not street data, so the
general-scene features transfer better.

**Q: Explain `ignore_mismatched_sizes=True` — which layers re-initialized and why?**
ADE20K's `decode_head.classifier` is `Conv2d(256, 150, 1)`; we need 3 classes. The flag tells HuggingFace to
**skip and randomly re-initialise only the mismatched layers** — the decode-head `classifier` **weight and
bias** — while loading the whole MiT-B0 encoder + MLP-fusion intact. The run log confirms exactly those two
tensors are re-initialised.

**Q: What is the MiT-B0 encoder and how does it differ from a standard ViT?**
The **Mix-Transformer** is a hierarchical ViT variant: (1) **overlapping** convolutional patch embeddings
(vs ViT's non-overlapping patches); (2) a **4-stage** pyramid producing multi-scale feature maps (vs ViT's
single scale); (3) **efficient self-attention** with spatial reduction (vs ViT's quadratic attention); and
no positional embeddings. Ideal for dense prediction.

**Q: What is the All-MLP decoder and how does it fuse multi-stage features?**
Each of the 4 encoder-stage features passes through a small **MLP** to a common width, is bilinearly
**upsampled** to 1/4 resolution, **concatenated**, and fused by another MLP into per-pixel logits — no heavy
conv decoder needed because the MiT already supplies multi-scale features.

**Q: HuggingFace loss vs your own loss — which and why?**
We compute **our own**. We call `model(pixel_values=x).logits` **without** `labels=`, so HF skips its
internal CE; we upsample the `H/4×W/4` logits to 256² and apply the **identical Dice + weighted-CE** from
NB1 — a fair benchmark needs the exact same loss across models.

**Q: How were images resized/padded, and where?**
None needed — data is natively **256×256** and the dataloader emits 256×256. SegFormer outputs 64×64 logits
(stride 4); we upsample them to 256² with `F.interpolate(bilinear)` inside our `forward_logits` wrapper
(our forward code — not inside the HF model, not in the dataloader), so loss/metrics use full resolution.

## 4. YOLOv26 Semantic (Notebook 3 — `seg-yolov26-semantic`)

**Q: Difference between `-sem` and `-seg`; why does the wrong one change the task?**
`-sem` = **semantic segmentation** (dense per-pixel class map — what we need). `-seg` = **instance
segmentation** (per-object masks + boxes + confidences). Ultralytics infers the task from the checkpoint
name, so `-seg` would train an instance model with box/instance outputs and mask-mAP metrics — a different
task entirely. We used `yolo26s-sem.pt`.

**Q: What does your `data.yaml` contain? Walk through each field.**
```yaml
path: /kaggle/working/yolo_ds   # dataset root
train: images/train             # train images (relative to path)
val:   images/val               # val images
test:  images/test              # held-out test images
masks_dir: masks                # folder with per-image masks (same filename stem)
names: {0: background, 1: liver, 2: tumor}   # class-id → name
```

**Q: What mask format does YOLOv26-sem expect and how did you verify it?**
Single-channel PNG, **pixel value = class ID** (0/1/2), **255 = ignore**. We generate masks from
`combined_label()` (values strictly `{0,1,2}`) and save with `Image.fromarray(uint8)` (single-channel L
PNG); we use **no** 255/ignore pixels. Verified via NB0's sanity grid (labels only ever `{0,1,2}`) and
YOLO's own validation reporting the 3 named classes.

**Q: What does `imgsz` control and how did you pick it?**
`imgsz` is the square resolution inputs are resized to. We set **256** to match the dataset's native
256×256, avoiding any rescale that would blur the already-tiny tumors (we deliberately did **not** use
YOLO's 1024 Cityscapes default).

**Q: How do you read back `result.semantic_mask.data` and convert it to mIoU?**
`model.predict(path, imgsz=256)[0].semantic_mask.data` → an `(H,W)` tensor of class IDs → NumPy → nearest-
resize to 256² if needed → accumulate into a 3×3 confusion matrix vs the GT (`np.bincount(3*gt + pred)`).
mIoU = `mean(TP_c/(TP_c+FP_c+FN_c))` — the **same `ConfMat` code** used for DeepLabV3/SegFormer, so all
three are measured identically.

**Q: What optimizer/LR schedule does Ultralytics use by default, and did you override any?**
Default `-sem`: `optimizer=auto` (SGD/AdamW auto-selected), `lr0=0.01` + **cosine** (`lrf=0.01`),
`momentum=0.937`, `weight_decay=5e-4`, `warmup_epochs=3`. We **kept** the optimizer/LR (documented) and only
overrode **augmentation**: `fliplr=0.5, flipud=0.5` and `hsv_h=hsv_s=hsv_v=0` (colour jitter off, since our 3
channels encode neighbouring slices, not colour).

## 5. Error Analysis & Final Comparison (Task H)

**Q: Which model achieved the best mIoU and why for this data?**
**DeepLabV3 (0.768).** (1) Our fairness fix (unifying weight-decay to 1e-2) most benefited the large,
previously under-regularised 42 M CNN, which had the capacity to exploit 2.5D context + oversampling;
(2) **ASPP's multi-scale receptive field** matches LiTS's large size variation. The small, already-
regularised SegFormer gained less — which flipped the ranking from earlier baselines.

**Q: Which class was hardest across all three, and why?**
**Tumor** (IoU 0.34–0.41). Statistical: ~0.4% of pixels (extreme imbalance). Visual: small, low-contrast
lesions inside the liver with similar intensity → all models leak **tumor→liver** (48–60%). Per-patient
tumor Dice varies from ~0 to ~0.85 (some lesions essentially missed).

**Q: Do the three models fail on the same images or different ones? What does that tell you?**
**Mostly the same, with model-specific tails.** Per-image IoU correlates at **r ≈ 0.91–0.93** (agreement on
which slices are hard → data-driven difficulty), but only **5 of each model's worst-30 slices are shared by
all three** (Jaccard 0.18–0.28). So the data ceiling dominates, while architecture shapes the worst-case
behaviour.

**Q: To improve the worst model, what specific change and why?**
Weakest is **YOLOv26-sem** on tumor. Highest-leverage changes: (1) **`imgsz` → 512** (its detector-lineage
downsampling destroys tiny lesions — higher resolution preserves them); (2) a **liver-ROI cascade** (segment
liver first, restrict tumor search there — removes distractors); (3) a **larger `-sem` variant**
(`yolo26m/l-sem`) for capacity. (1)+(2) directly target the tiny-tumor failure mode.

---

## Appendix — theory quick-reference (§6.2 support)

- **IoU** = |A∩B|/|A∪B|; **Dice** = 2|A∩B|/(|A|+|B|) = pixel-level F1; mIoU beats pixel accuracy on
  imbalanced data because background inflates accuracy but not IoU.
- **Focal loss** down-weights easy pixels to focus on hard/rare ones; class weights `= median(freq)/freq_c`.
- **Encoder–decoder:** encoder compresses to semantic features (loses detail); decoder recovers resolution;
  **skip connections** (U-Net) re-inject high-res detail lost in downsampling.
- **Atrous/dilated convolution** grows the receptive field with no extra parameters or downsampling;
  **output stride** = input/feature ratio (lower = finer masks, more compute).
- **Data leakage** here = a patient's near-identical slices split across train/test → we split by volume.
- **Overfitting** shows as train loss ↓ while val loss ↑ (see our v2 ablation); we mitigate with augmentation,
  weight decay, and best-checkpoint selection.
- **Transfer learning** beats random init because low-level edge/texture filters are universal — faster
  convergence and better generalisation on a small dataset.
