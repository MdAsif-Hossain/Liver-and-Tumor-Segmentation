# Universal Run Guide — how to run every notebook, in order

Single source of truth for **what to attach, which GPU, and what order**. Everything runs on **Kaggle** (the local
repo is for authoring only). For every notebook: **Run All** → confirm outputs visible → set **Public** → **Save
Version** → **publish the output as a Kaggle Dataset** if a later notebook needs it.

Legend: 🖥️ GPU · 📎 attach · 📤 publish output as dataset.

---

## Project 1 — Assignment 1 (Part A): supervised benchmark *(complete)*

🖥️ **T4 ×2** (or P100) · Internet **On**.

| ▶️ | Notebook | 📎 Attach |
|---|---|---|
| 1 | `eda-and-data-prep` | LiTS 256×256 → 📤 publish (`split.json`, `class_weights.json`, `manifest.csv`) |
| 2 | `seg-deeplabv3` | LiTS + NB0 output |
| 3 | `seg-segformer-b0` | LiTS + NB0 output |
| 4 | `seg-yolov26-semantic` | LiTS + NB0 output |

**LiTS:** `sodko3/lits-dataset-liver-and-tumor-segmentation-256x256`.
**Critical:** notebooks 2–4 attach the **same NB0 output** so the split is byte-identical.

---

## Project 2 — Assignment 2 (Part B): self-supervised *(6 notebooks, code-complete)*

🖥️ **T4 ×2 for every notebook** · Internet **On** (ImageNet / DINOv2 weights).
**Why T4 and not P100:** all notebooks train under **AMP (fp16)**. T4 has **tensor cores** (~65 TFLOPS fp16);
P100 has **none**, so AMP barely helps it. SimCLR and MAE additionally wrap the SSL model in `nn.DataParallel`,
so they use **both** T4s (set `USE_DATA_PARALLEL = False` to disable). BYOL and DINOv2 stay single-GPU on purpose —
their EMA-target/BatchNorm semantics are sensitive to per-device batch splitting.
Each method notebook is standalone and covers the PDF's internal order **(1)–(8)**: setup+version pins → load
splits → **50-ep SSL pretrain** (pretext curve + per-epoch time) → decoder attach → **50-ep fine-tune** on the
labelled *val* split (monitored on *test*) → Task E metrics → **Task F error analysis** → `results.json`.

| ▶️ | Notebook | 📎 Attach | Produces |
|---|---|---|---|
| 1 | `lits-ssl-data-prep` | LiTS + **Part-A NB0 output** | 📤 `partB_split_roles.json`, sanity grid |
| 2 | `lits-simclr` | LiTS + **data-prep output** | 📤 `results.json`, `simclr_per_image_iou.csv`, curves, Task-F grids |
| 3 | `lits-byol` | LiTS + data-prep output | 📤 same for BYOL |
| 4 | `lits-mae` | LiTS + data-prep output | 📤 same for MAE |
| 5 | `lits-dinov2` | LiTS + data-prep output | 📤 same + **frozen-vs-fine-tuned** comparison |
| 6 | `lits-partB-final-comparison` | data-prep + **all 4 method outputs** + **Part-A `results.json`** | label-efficiency table/charts + consolidated Task F |

**Run notebook 2 (SimCLR) first and confirm it completes** — BYOL/MAE share its exact structure, so any bug
found there applies to all three.

**Timing / quota.** Each method is 50 SSL + 50 fine-tune epochs. Run methods on **different days** if you hit the
weekly GPU quota. Every notebook prints **seconds/epoch**; per the PDF, if either stage exceeds **10 min/epoch**,
split that method into a `-pretrain` (📤 saves encoder) + `-downstream` pair. **DINOv2 (ViT + multi-crop) is the
most likely to need this** — check its printed epoch time first.

**Notes.** DINOv2 runs at **224×224** (ViT-S/14 needs a 14-divisible size); the ResNet methods use 256. DINOv2's
self-distillation stage has a **fallback**: if it can't run, it drops to released DINOv2 weights and still
completes the segmentation stage.

### 🔬 EXTRA cells
Each method notebook ends with a clearly-marked **EXTRA — additional research visualisations** section
(t-SNE feature progression · per-patient variance & lesion-size sensitivity · confidence calibration). These are
**not required by the assignment** and can be skipped; everything above that banner is the graded work.

---

## Project 3 — Research: dehazing via Dark Channel Prior

🖥️ **None** (DCP is CPU, seconds/image) — GPU **only** for the optional learned reference. Internet **On** only
for that optional baseline.

| ▶️ | Notebook | 📎 Attach |
|---|---|---|
| 1 | `dehaze-dcp-pipeline` | RESIDE-SOTS or O-HAZE *(else synthetic fallback runs)* |
| 2 | `dehaze-ablation-eval` | same dataset |

**Datasets:** search Kaggle for "RESIDE SOTS" (synthetic, perfectly paired — **primary**), "O-HAZE"/"I-HAZE"
(real pairs), "Dense-Haze NTIRE 2019". With none attached both notebooks **auto-generate synthetic haze**
(`I = J·t + A(1−t)`) so they run anywhere — but attach RESIDE for reportable numbers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `assert roles_paths` fails | The Part-B data-prep output isn't attached (needs `partB_split_roles.json`) |
| `split.json` not found (data-prep) | Attach the **Part-A NB0 output** dataset |
| CUDA probe prints "failed → CPU" | Kaggle gave a bad GPU — restart the session / re-pick the accelerator |
| OOM during SSL | Lower `CONFIG["ssl_batch"]` (96 → 48 → 32) |
| Epoch > 10 min | Split that method into pretrain + downstream notebooks (PDF allows it) |
| Final comparison shows "No SSL results" | Attach every method notebook's output (each has its own `results.json`) |

## Pre-submission checklist
- [ ] All notebooks **Run All**, outputs visible, **Public**, **Saved**.
- [ ] Outputs feeding later notebooks published as Datasets and attached downstream.
- [ ] Part B: **6 Kaggle links** (or more if a method was split) listed one-per-line in the Classroom post.
- [ ] Reports exported to PDF (Part B: two-column LaTeX with the mandatory Insights section).
