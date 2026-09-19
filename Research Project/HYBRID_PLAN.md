# Hybrid Plan — Where should learning enter a physics-based dehazing pipeline?
## Paper 2 (AI): stage-wise hybridisation of the Dark Channel Prior, with a prior-as-loss failure test

**Course:** CSE 348 / 438 — Digital Image Processing · **Group 01**, Dept. of CSE, East West University
**Members:** Md. Asif Hossain (2022-3-60-007) · Nabil Subhan (2022-3-60-063) · K M Nudar (2022-3-60-234)
**Builds on:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md) (paper 1, classical study) — same data, same split, same metrics, same statistics.
**Notebook:** `dehaze-hybrid-stages.ipynb` (NB 4) · **Report:** [report/report_hybrid.tex](report/report_hybrid.tex)

> **Order of work.** Paper 1 (classical, NB 1–3) is complete and is submitted first. Paper 2 reuses its protocol and its
> tuned DCP configuration; nothing in NB 1–3 changes.

---

## 1. Research question & hypotheses

**RQ.** The DCP pipeline has five classical stages. If a small learned module replaces **one stage at a time** — the
remaining stages kept classical and fixed — *which stage benefits most from learning*, and does the answer change with
haze density and on real (vs synthetic) haze?

| Hypothesis | Prediction | Test |
|---|---|---|
| **H4 — transmission is the learnable stage** | Replacing transmission refinement with a learned refiner yields the largest paired gain over the tuned DCP; replacing recovery yields the smallest. | learning gain per stage (paired Δ PSNR/SSIM vs tuned DCP, 95 % CI, Wilcoxon), ranked |
| **H5 — the prior as a loss inherits the prior's failure** *(paper 1's H2, learned)* | A refiner trained **only with the dark-channel prior as its loss** (no ground truth) does *not* reduce error in intrinsically bright regions — it can worsen it — whereas the same refiner trained with ground truth does. | error inside bright-region masks vs elsewhere, per arm; paired Δ of bright-region error vs the classical guided filter |
| **H6 — hybrids transfer to real haze better than end-to-end learning** | Because the physics scaffold is fixed, single-stage hybrids trained on synthetic RESIDE-ITS lose less of their gain on real haze (O-HAZE / I-HAZE / Dense-Haze) than the all-learned model (AOD-Net) trained with the same data and budget. | gain on synthetic sets vs gain on real sets, per arm (sim-to-real drop) |

## 2. Arms (every arm is evaluated on the identical test split of paper 1)

| Arm | What is learned | Classical stages kept | Training signal |
|---|---|---|---|
| DCP baseline | — | all (He et al. defaults) | — |
| **DCP tuned** (reference) | — | all (NB 2's tuned configuration) | — |
| **H-A** | atmospheric light: residual correction ΔA to the classical estimate | dark channel, transmission, guided filter, recovery | L1 to ground truth *through* the fixed classical stages |
| **H-t (sup)** | transmission refinement: residual Δt on the guided-filter output | dark channel, A, transmission, recovery | L1 to ground truth through the fixed recovery |
| **H-t (prior)** | same network as H-t (sup) | same | **no ground truth**: dark-channel-of-output loss + anchor to the guided estimate + TV; trained on *hazy images only*, including real ones |
| **H-J** | recovery / colour correction: residual ΔJ on the DCP output | everything up to recovery | L1 to ground truth |
| **H-all** | H-A + H-t (sup) + H-J composed | none learned jointly — modules trained separately and plugged in | (composition only) |
| **AOD-Net** | everything (end-to-end) | none | L1 to ground truth, same data and step budget |
| hazy input, CLAHE | — | — | model-free controls |

Design rule: **every learned module is a residual correction of its classical counterpart** (`A = A_cls + ΔA`,
`t = t_guided + Δt`, `J = J_dcp + ΔJ`), bounded by `tanh`, so an untrained module reproduces the classical pipeline and the
gain is attributable to learning. Modules are tiny (≈ 20–60 k parameters) so the comparison is about *where* learning
enters, not about capacity.

Training uses differentiable PyTorch re-implementations of the classical stages (erosion = min-pooling, guided filter =
box filters, recovery = the closed-form equation) so gradients flow through the fixed stages; evaluation runs the
*NumPy* classical stages of paper 1 with only the learned module in PyTorch, so hybrids are exact drop-ins.

## 3. Data & training protocol

* **Evaluation:** paper 1's records, scene-grouped tune/test split and fingerprint (NB 4 reads `dehaze_results.json`
  from NB 2 and refuses to run on a different split).
* **Training pairs:** RESIDE-ITS (or OTS) attached on Kaggle — auto-detected and excluded from evaluation (≤ 1,500 pairs
  cached in memory at ≤ 320 px). Fallback: synthetic haze rendered from *tune-split* ground truths (test scenes unseen).
* **Unsupervised training (H-t prior):** hazy side of the training pairs **plus the real hazy images of the tune split**
  (no ground truth is touched) — the one arm that can see real haze during training.
* **Budget:** identical for every learned arm — `STEPS` gradient steps (3,000 on GPU), batch 8, 224-px crops, Adam
  1e-3 with cosine decay; **3 seeds** on GPU (1 on CPU). Per-image scores are averaged over seeds before the paired
  statistics; seed spread is reported.
* **Metrics & statistics:** exactly paper 1 (PSNR, SSIM, CIEDE2000, Hautière e / r̄; paired bootstrap CIs, Wilcoxon;
  density terciles; bright-region masks from the clear image).
* **Cost:** parameter count and ms/image per arm (classical arms on CPU, learned modules on the device used).

## 4. Analyses & figures (all written by NB 4 with `numbers_hybrid.tex` macros)

| Analysis | Output |
|---|---|
| Learning gain per stage (H4) | `tab_hybrid_main`, `fig_hybrid_gain_per_stage` |
| By density stratum | `tab_hybrid_by_stratum`, `fig_hybrid_by_density` |
| Bright-region test (H5) | `tab_hybrid_bright`, `fig_hybrid_bright`, `fig_hybrid_tmaps` (raw / guided / learned-sup / learned-prior transmission on a bright-region image) |
| Sim-to-real drop (H6) | `tab_hybrid_by_dataset`, `tab_hybrid_simreal` |
| Cost | `tab_hybrid_cost` (params, ms/img, training minutes) |
| Training curves, qualitative grid | `fig_hybrid_training`, `fig_hybrid_qualitative` |

## 5. Compute (Kaggle)

| Step | Where | Time |
|---|---|---|
| Train 5 learned arms × 3 seeds × 3,000 steps | T4 | ≈ 30–45 min |
| Evaluate 9 arms × 3 seeds on ~1,100 test images | T4 (CPU for classical stages) | ≈ 45–60 min |
| **Total NB 4** | | **≈ 1.5–2 h**, one session |

## 6. Timeline

| When | What |
|---|---|
| Week 1 | Kaggle run of NB 1–3 (paper 1) with the real benchmarks; write paper 1's Insights; submit paper 1 |
| Week 2 | Attach RESIDE-ITS; run NB 4; check the inventory and the fingerprint match |
| Week 3 | Compile `report_hybrid.tex`; write Discussion; decide the headline from H4–H6 |
| Week 4 | Internal review with REPORT_GUIDE Phase 4; submit paper 2 |

## 7. Risks

| Risk | Mitigation |
|---|---|
| The prior-as-loss refiner collapses to its anchor (learns nothing) or over-darkens skies | Both are *results* for H5 and are reported as such; the supervised twin isolates the effect of the training signal |
| ITS not attached | Synthetic-from-tune-GT fallback runs; the report states the route (macro `hybTrainSource`) |
| GPU quota | Every arm trains in minutes; `SEEDS` and `STEPS` are single-line settings; CPU fallback exists |
| Train/eval mismatch between PyTorch and NumPy classical stages (padding) | Modules are residual on the classical output, so the mismatch is a small input perturbation; documented in Threats to Validity |
| "It is just DehazeNet/MSCNN again" | Those learn *t* end-to-end; this study keeps the scaffold fixed and asks *which* stage, with identical budgets, statistics and a real-haze transfer test |

## 8. Venue framing

Image restoration / low-level vision track (ICIP, ICASSP, WACV, VCIP; MICCAI-style venues are not the fit). Title
working draft: *"Where Should Learning Enter a Physics-Based Dehazing Pipeline? A Stage-Wise Study of the Dark Channel
Prior."*
