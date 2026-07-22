# CSE 438 — Part B Execution Plan
## Self-Supervised Learning for Semantic Segmentation (SimCLR · BYOL · MAE · DINOv2)

**Course:** CSE 438 — Digital Image Processing · Assignment **Part B** · Summer 2026
**Group:** 01 · Dept. of CSE, East West University
**Members:** Md. Asif Hossain (2022-3-60-007) · Nabil Subhan (2022-3-60-063) · K M Nudar (2022-3-60-234)
**Dataset:** LiTS 256×256 (liver + tumor CT) — *same as Part A*
**Deadline:** Sec. 3 → 17.08.2026 · Sec. 4 → 18.08.2026 · **Prerequisite: Part A submitted & evaluated**

---

## 0. The central question

> **How much of the Part-A fully-supervised result can each SSL method recover using only ~19% of the labels?**

Part A trained supervised on **13,447 labelled** slices and reached **DeepLabV3 test mIoU = 0.768**.
Part B pretrains an encoder with **no labels**, then fine-tunes on only the **2,553-slice validation split**
and measures how close it gets. Label ratio = **2,553 / 13,447 ≈ 0.19 (≈ 1/5)**.

---

## 1. What we inherit from Part A (reuse, do NOT rebuild)

| Artifact | Source | Role in Part B |
|---|---|---|
| `split.json` (92/20/19 volumes → 13,447/2,553/3,158 slices) | Part A NB0 Kaggle dataset | Loaded **unmodified**; verify byte-identical (hash check) |
| `manifest.csv`, `class_weights.json` | Part A NB0 | Per-slice `has_tumor` flags, CE weights `[0.3,1,6]` |
| **DeepLabV3 ASPP decoder head** (Part A winner) | Part A NB1 code | Re-attached on top of every SSL encoder (Task C) |
| Part A val augmentation `A.Compose([...])` | Part A NB0 | Labelled fine-tuning loader (synchronized image+mask) |
| `ConfMat` + metric functions (mIoU, per-class IoU, Dice, tumor-F2) | Part A NB1/NB3 | Identical Task E metric set |
| `results.json` baseline (DeepLabV3 0.768 mIoU, tumor IoU 0.407 …) | Part A NB3 | Label-efficiency comparison table |
| 2.5D 3-channel input `[i-1, i, i+1]` | Part A NB0 `load_25d` | Maps naturally to the 3 RGB input channels of ImageNet SSL checkpoints |

**Split → SSL role remapping (Section 2 of the brief):**

| Part-A split | Slices | New Part-B role | Masks? |
|---|---|---|---|
| **Train** | 13,447 | Unlabelled SSL pretraining corpus (Task B) | discarded |
| **Validation** | 2,553 | Labelled fine-tuning data (Task D) | **used** |
| **Test** | 3,158 | Downstream monitoring **and** final eval (Tasks D & E) — *double role* | used at eval |

> ⚠️ **Test-split double role** must be disclosed in report §6.3 + Task-D markdown: the test set is touched
> during checkpoint selection, so reported metrics are **mildly optimistic** vs. a truly-held-out set. The
> "correct" fix (a 4th held-out slice) is infeasible given the small labelled set. This is a viva question.

**Do NOT** improve Part A's tumor score — it would make the baseline a moving target. Only optional
cosmetic doc-carry-forward: oversampling wording "~38%→~71%" and hedge "wins **pooled** tumor metrics".

---

## 2. Locked design decisions

> **These are locked to the course's own lab notebooks** (`Lab_practice2/nb-02..06-*-deeplabv3`), which are the
> starter templates the PDF says to "adapt to your assigned dataset." Their decisive convention: **SimCLR, BYOL,
> and MAE all use a ResNet-50 encoder** so it drops straight into DeepLabV3's ResNet-50 backbone — **only DINOv2
> uses a real ViT** and therefore the token→grid adapter. This is why the PDF's "typical backbone" (ViT for MAE)
> is superseded by the lab's ResNet-50 CNN-MAE: it makes the DeepLabV3 transfer methodologically valid.

| Decision | Choice | Justification / viva point |
|---|---|---|
| **Pretraining path** | **ImageNet-init ResNet-50 (or SSL ckpt) → continue SSL pretraining 50 epochs** (path 1) | labs pretrain 10 ep from scratch; we do 50 ep from ImageNet init per PDF |
| **SimCLR** | ResNet-50 + MLP projector (2048→512→128); NT-Xent, τ≈0.2 | report τ; loss in **float32** even under AMP |
| **BYOL** | ResNet-50 online+EMA target; projector 2048→2048→256(BN), predictor 256→512→256 | report EMA momentum (0.996→1 cosine); symmetric neg-cosine loss (float32) |
| **MAE** | **ResNet-50 CNN-MAE** + transposed-conv recon decoder (7→224); 75% patch-mask, 16px; masked L1+0.2·L2 | matches lab; **no adapter** — transfers straight to DeepLabV3 backbone |
| **DINOv2** | ViT-S/14 (`facebook/dinov2-small`, HF `Dinov2Model`) | the **only** ViT; report patch 14, 224/14=16 grid |
| **Decoder — SimCLR/BYOL/MAE** | **Full torchvision `deeplabv3_resnet50(aux_loss=True)`** = literally our Part A NB1 model | `backbone.load_state_dict(encoder.state_dict(), strict=False)` |
| **Decoder — DINOv2** | Reimplement the **ASPP head** on ViT tokens: drop CLS, `(B,N,C)→(B,C,16,16)`, ASPP rates **(3,6,9)** + classifier → bilinear upsample | dilation rates small because grid is only 16×16 |
| **Frozen vs fine-tuned** | **DINOv2 carries the required experiment**: warm up ASPP head with encoder **frozen** (2 ep), then unfreeze (encoder LR 1e-5, head LR 2e-4). Full fine-tune the other 3 | brief requires both-for-one; lab's warm-up staging is exactly this |
| **Loss / optim / metrics** | Reuse Part A: Dice + CE`[0.3,1,6]` (+0.4·aux for the CNN methods), AdamW warmup→cosine, `ConfMat` 3-class | **NUM_CLASSES=3** (bg/liver/tumor), not the lab's binary |
| **Checkpoint metric** | Monitor on **test split** (double role), select best **test mIoU** | PDF §3.4 wording; note tumor-F2 as secondary |

**Universal lab scaffolding to reuse in every method notebook** (all five labs share it):
CUDA-kernel probe before selecting GPU → AMP only if probe passes → **SSL/contrastive/regression loss always in
float32** (autocast disabled) to avoid fp16 overflow → two-stage (SSL pretrain → transfer → fine-tune) → CE +
foreground-Dice objective → per-epoch history CSV + curves → **t-SNE feature-space progression** (before-SSL /
after-SSL / after-fine-tune) + qualitative panels (image/GT/prob/pred/overlay/error) + spatial-response maps →
save `test_metrics.csv`, checkpoints, `experiment_summary.json`.

**What we must change vs the labs** (they target a different dataset/scale):
1. **Dataset** → LiTS **3-class** (bg/liver/tumor) via our NB0 label fusion, not brain-tumor COCO binary.
2. **Split** → reuse Part-A leakage-safe split with SSL role remap; **fine-tune on the 2,553 val slices only**, not the full train split (the labs fine-tune on their whole train set).
3. **Epochs** → **50 + 50**, not 10 + 10.
4. **Init** → ImageNet/SSL checkpoint, not `weights=None` from scratch.
5. **Metrics** → Part-A set (mIoU, per-class IoU, pixel-acc, Dice, confusion matrix) + label-efficiency table vs Part-A 0.768.
6. **Input** → keep Part-A **2.5D** 3-channel.

---

## 3. Notebook structure (6–10 notebooks)

Split ViT methods (slower/epoch) into pretrain+downstream pairs **only if** per-epoch time > 10 min.
Planned layout (**decide split empirically from measured epoch time — report it either way**):

| # | Notebook | Contents |
|---|---|---|
| 0 | `lits-ssl-data-prep` | Task A: load Part-A split unmodified + hash-verify, 3 role loaders, per-method aug, sanity grid |
| 1 | `lits-simclr` (combined) | ResNet-50 → SimCLR pretrain 50ep → ASPP → fine-tune 50ep → test → error analysis |
| 2 | `lits-byol` (combined) | same, BYOL |
| 3a/3b | `lits-mae-pretrain` / `lits-mae-downstream` | ViT-B: split if >10 min/epoch |
| 4a/4b | `lits-dinov2-pretrain` / `lits-dinov2-downstream` | ViT-S/14: split if >10 min/epoch; also runs frozen-vs-finetune |
| 5 | `lits-partB-final-comparison` | pull all `results.json` + Part-A baseline → label-efficiency table + chart + consolidated error analysis |

**Internal cell order for every method notebook (brief §4):**
1. setup + version-pinned imports
2. load shared train(unlabelled)/val(labelled)/test(eval) splits from data-prep
3. SSL pretraining stage — 50 epochs, **pretext-loss curve**, per-epoch time (save encoder ckpt if split)
4. decoder attachment + frozen/fine-tuned config (load ckpt first if split)
5. fine-tuning config cell + loop — 50 epochs on val, **per-epoch monitor loss/mIoU on test**, per-epoch time
6. final test-set eval (Task E metrics)
7. error-analysis visualizations (Task F)
8. markdown summary → append to shared `results.json`

---

## 4. Task-by-task checklist (mark weights)

- **Task A — Data Prep (8):** load Part-A file lists unmodified; save index artifacts; train loader = augmented
  **views only, no masks**; val loader = Part-A synced image+mask; test loader = **deterministic, no aug**;
  sanity grid: (a) two augmented views of one train image, (b) val image + mask overlay; report exact counts.
- **Task B — SSL Pretraining ×4 (24):** each method 50 epochs on train split; checkpoint-continuation;
  pretext-loss curve + wall-clock/epoch; adapt starter notebooks (don't run unmodified).
- **Task C — Decoder + Adapter (8):** DeepLabV3 ASPP head on each encoder; document ViT token→grid reshape;
  state frozen vs fine-tuned per method.
- **Task D — Fine-tuning ×4 (20):** fine-tune on **val split only**, exactly 50 epochs; **report the epoch-50
  downstream mIoU even if the curve is still trending upward** (50 is fixed, not a floor); monitor+checkpoint
  on test split; **single visible config cell listing**: optimizer · LR (separate encoder vs decoder if used) ·
  schedule · batch size · input resolution · loss · frozen-vs-fine-tuned flag · **measured per-epoch time (both
  stages)** · total training time; plot fine-tune loss (val) **alongside** test monitor loss/mIoU.
- **Task E — Metrics + Label-Efficiency (15):** mIoU, per-class IoU, pixel-acc, Dice, confusion matrix on test;
  **label-efficiency table**: each SSL method's test mIoU (on 2,553 labels) vs Part-A DeepLabV3 (on 13,447);
  discuss which method closes the gap best with how much less data.
- **Task F — Error Analysis (10):** worst per-image-IoU test slices vs GT per method; do errors land on the
  same tumor/edge cases as Part A supervised, or different? hypothesize why.
- **Report (15):** two-column LaTeX, no lit-review, **mandatory Insights §6.9** (original analysis, not restated).

---

## 5. Per-method pretext + augmentation notes (viva-critical)

| Method | Pretext | Key augs (adapt for CT) | Report |
|---|---|---|---|
| **SimCLR** | contrastive NT-Xent: pull 2 views of same image together, push others apart | RandomResizedCrop, flip, light color-jitter/blur (grayscale CT → soften color aug) | temperature τ, batch size (negatives) |
| **BYOL** | online predicts target's rep of a differently-augmented view; no negatives | asymmetric aug pair; stop-grad + EMA target prevents collapse | EMA decay |
| **MAE** | mask ~75% patches, reconstruct pixels; encoder sees visible patches only | minimal: RandomResizedCrop + flip (masking *is* the pretext) | masking ratio, patch size |
| **DINOv2** | self-distillation: student matches EMA-teacher over multi-crop | 2 global + N local crops | patch 14, #crops, centering/sharpening |

**Insights §6.9 angles (write your own — AI can't):** does masked-reconstruction (MAE) or self-distillation
(DINOv2) suit low-texture CT better than contrastive (SimCLR/BYOL)? did freezing help on the tiny 2,553-label
set? which method recovers tumor IoU best relative to Part A's 0.407?

---

## 6. Compute & sequencing

- **Path 1 (checkpoint-continuation)** everywhere → keeps 50-ep pretrain feasible on T4/P100.
- Measure per-epoch time in each config cell; **split MAE/DINOv2 if >10 min/epoch** (save encoder as Kaggle dataset).
- **Build order:** data-prep → SimCLR (ResNet, validates the whole pipeline cheapest) → BYOL → MAE → DINOv2 → final.
- Each method appends to a shared `results.json`; final notebook + Part-A `results.json` → one comparison table/chart.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ViT patch-token → ASPP shape mismatch | explicit reshape adapter + assert `H*W == N`; unit-test on one batch first |
| 50-ep ViT pretrain hits Kaggle 9h session limit | split into pretrain/downstream notebooks, checkpoint as dataset |
| Grayscale-CT breaks SimCLR color aug | reduce color-jitter strength; keep geometric augs; note in report |
| SSL checkpoint expects RGB/224 | 2.5D is already 3-channel; resize/interp to encoder's native size, document |
| Test-split double-role looks like leakage | disclose in §6.3 + Task-D markdown; explain metric inflation at viva |

## 8. Report → deliverable map (LaTeX two-column, 4–6 pp)

6.1 title/authors/abstract · 6.2 intro + one-para Part-A recap · 6.3 dataset & split roles + double-role caveat ·
6.4 methodology (4 pretext tasks, backbones, adapter) · 6.5 fine-tune setup + curves · 6.6 results table+chart ·
6.7 error analysis vs Part A · 6.8 label-efficiency ranking + exact ratio (0.19) · **6.9 Insights (mandatory)** ·
6.10 conclusion + refs (SimCLR/BYOL/MAE/DINOv2 papers).

---

## 9. AI Usage Policy, submission & integrity — hard compliance rules (PDF §5, §7, §11)

**AI Usage Policy (§7) — this governs how the AI assistant may help:**
- ✅ AI **allowed** for: boilerplate code, debugging error messages, understanding library APIs.
- ❌ AI **forbidden** for: the **Insights section (§6.9)**, the **error-analysis reasoning (Task F / §6.7)**, and
  **viva answers** — these must be the group's own words and understanding. *The assistant will scaffold code and
  list the questions, but the group writes all Insights + error-analysis interpretation themselves.*
- 📝 Every **non-trivial AI-assisted code block** must carry a one-line disclosure comment in the notebook
  (e.g. `# scaffolded/debugged with AI assistance`). Undisclosed AI blocks are what gets penalized at viva.
- 🎓 Every member must be able to **explain any line of any of the 6–10 notebooks** and any report claim;
  inability reduces that member's individual viva mark regardless of output quality.
- 📚 Cite SimCLR/BYOL/MAE/DINOv2 papers + library docs **in markdown cells and in the report** (§11).
- 🚫 Near-identical notebooks/reports/numbers across groups = integrity investigation → keep hyperparameters,
  augmentation choices, and Insights genuinely our own.

**Viva-prep artifact (policy-safe):** unlike Part A's `CODING_QUESTIONS.md`, we will **not** pre-author viva/Insights
answers. Instead prepare a *question → notebook-cell pointer* checklist (§9.1 coding + §9.2 theory questions mapped
to the exact cell that answers each) so each member studies from their own code. Answers stay theirs.

**Submission post format (§5) — one line per notebook, single Google Classroom post:**
```
Group: 01        Section: <3 or 4>        Dataset: LiTS 256x256 (same as Part A)
Data Prep:  <kaggle link>
SimCLR (combined | pretrain+downstream):  <link(s)>
BYOL   (combined | pretrain+downstream):  <link(s)>
MAE    (combined | pretrain+downstream):  <link(s)>
DINOv2 (combined | pretrain+downstream):  <link(s)>
Final Comparison:  <kaggle link>
Report PDF:  <attached / drive link>
```
Before posting: every notebook **run end-to-end with outputs visible**, set **Public**, report exported to **PDF**,
submitted by **your section's deadline** (Sec.3 17.08.2026 / Sec.4 18.08.2026).

---

## 10. Immediate next steps

1. Confirm Part A `split.json`/`results.json` Kaggle dataset is attachable (it is — NB1–3 used it).
2. Build **NB0 data-prep** first: load split unmodified, hash-verify no re-shuffle, wire 3 role loaders + sanity grid.
3. Then SimCLR (cheapest full end-to-end) to validate encoder→ASPP→fine-tune→metrics plumbing before scaling to ViT methods.
