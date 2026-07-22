# CSE 348 / 438 — Master Roadmap (3 phases)

**Group 01 · Dept. of CSE, East West University** — Md. Asif Hossain (2022-3-60-007) · Nabil Subhan
(2022-3-60-063) · K M Nudar (2022-3-60-234)

This roadmap sequences three bodies of work: **(1)** finalise Assignment 1 (Part A), **(2)** drive Assignment 2
(Part B) to a perfect submission, **(3)** a standalone research project from the Project Idea Bank.

---

## Phase 1 — Assignment 1 (Part A): fix & finalise

**Verdict as your research lead: Part A needs NO model retuning. It needs two documentation fixes (now done).**

### Why no retuning
- **Methodologically sound.** Leakage-safe volume-grouped split, fair identical-recipe benchmark, defensible
  loss/sampling/checkpoint choices. There is no correctness bug that a retrain would fix.
- **The weak tumor score is a ceiling, not a tuning miss.** ~45% tumor recall is the expected limit of 2D,
  pre-windowed 8-bit-PNG segmentation. Only an architectural change (liver→tumor cascade, true 3D / HU windows)
  moves it materially — and that is **Part-2 / research scope**, not a tune of Part A.
- **Retraining Part A now is actively harmful to Part B.** Part B's whole thesis is a label-efficiency comparison
  against a **fixed** Part-A baseline (DeepLabV3 mIoU **0.768**). Re-baselining mid-stream breaks that comparison.

### Fixes applied (this pass)
- ✅ README TL;DR: "wins **every** tumor metric" → "wins mIoU and the **pooled** tumor metrics (IoU/Dice/sensitivity)"
  — honest because YOLO may edge DeepLabV3 on *mean-per-patient* tumor Dice.
- ✅ README §2: oversampling "~12% → ~45%" → "**~38.6% → ~71%**" (correct math: 4·p/(1+3·p), p≈0.386, filtered set).

### Carry-forward (do on next Kaggle edit; local `.ipynb` comments don't reflect until re-run)
- [ ] Same one-line oversampling-comment fix in `seg-deeplabv3.ipynb` and `seg-segformer-b0.ipynb` config cells.
- [ ] *(Optional, cheap, no retrain)* threshold-sweep operating-point analysis — **defer**: it would change the
  reported baseline that Part B cites. Only add if we freeze Part B's baseline reference to the argmax numbers.

**Status: Phase 1 complete for submission purposes.**

---

## Phase 2 — Assignment 2 (Part B): path to a *perfect* submission

Full technical plan lives in [Assignment 2 (Part B)/PART_B_PLAN.md](Assignment 2 (Part B)/PART_B_PLAN.md).
"Perfect" = every PDF clause met, every notebook runs clean on Kaggle with outputs, LaTeX report with a real
Insights section, viva-ready. Execution order and quality bar:

### Build order (each validated before the next)
1. ✅ **NB0 `lits-ssl-data-prep`** — built & syntax-clean (Task A: split reuse + role remap + sanity grid).
2. **`lits-simclr`** — cheapest full end-to-end; **validates the whole ResNet-50 → DeepLabV3 → fine-tune → 3-class
   metrics pipeline** before anything ViT. Once this is green, BYOL and MAE are near-copies.
3. **`lits-byol`** — swap SimCLR head for online/EMA-target; same DeepLabV3 transfer.
4. **`lits-mae`** — ResNet-50 CNN-MAE (transposed-conv decoder); same transfer. *(split pretrain/downstream only if >10 min/epoch)*
5. **`lits-dinov2`** — the one ViT: token→grid adapter + reimplemented ASPP head; carries the **frozen-vs-fine-tuned** experiment.
6. **`lits-partB-final-comparison`** — label-efficiency table + chart vs Part-A 0.768; consolidated error analysis.

### Quality bar per notebook (the "perfection" checklist)
- [ ] Group/course header table + version-pinned imports (match Part A style).
- [ ] Loads NB0 role artifacts; re-verifies split fingerprints (no re-shuffle).
- [ ] 50-epoch SSL pretrain, ImageNet-init, **float32 SSL loss**, pretext-loss curve + per-epoch wall-clock.
- [ ] 50-epoch fine-tune on **val split only**, monitored on **test split**, single visible config cell.
- [ ] Part-A 3-class metric set (`ConfMat`: mIoU, per-class IoU, pixel-acc, Dice) + confusion matrix.
- [ ] Task-F worst-IoU error grids + comparison to Part-A failure modes.
- [ ] Appends to shared `results.json`; run end-to-end with outputs; set **Public**.

### Report (LaTeX two-column, 4–6 pp) — §6.1–6.10, **mandatory Insights §6.9**, no lit-review.
### Known risk to close early: ViT token→ASPP shape (assert `H·W==N` on one batch first).

**Status: NB0 done; NB1 (SimCLR) is the next build.**

---

## Phase 3 — Research project (from the Project Idea Bank)

New folder created: **[Research Project/](Research Project/)** — the detailed plan lands here once the topic is chosen.

### Framing (as expert scientist)
The Idea Bank is **classical DIP** territory (Otsu, watershed, Hough, Wiener, dark-channel prior, K-means,
morphology). The group's edge — a rigorous, reusable **segmentation evaluation harness** (Dice/IoU/`ConfMat`,
leakage-safe protocol, error analysis) and hands-on **classical-vs-learned** perspective from Parts A/B — makes
the strongest research angle a **quantitative, reproducible study**, not a demo. Two viable directions:

- **Segmentation-family topic** → maximal reuse of the Parts A/B evaluation harness; natural research question
  *"how close can an engineered classical pipeline get to a learned model, and precisely where does it break?"*
- **Enhancement/Restoration topic** → non-medical (matches the earlier "next project is not medical data" intent),
  quantitative via PSNR/SSIM on standard benchmarks; a clean classical-technique study.

### Research-plan template (to be filled per chosen topic)
1. **Problem & research question** (a testable hypothesis, not a demo goal)
2. **Dataset(s)** with public GT + license; train/val/test or eval protocol
3. **Classical method pipeline** (each DIP operator justified) + baseline(s)
4. **Evaluation protocol & metrics** (reuse Parts A/B harness where applicable)
5. **Ablations / parameter studies** (the research substance)
6. **Comparison** (classical vs learned, or vs literature)
7. **Insights & failure analysis**, **reproducibility**, **threats to validity**
8. **Deliverables** (notebook(s) + report/figures) & timeline

### Topic — DECIDED by 3-advisor council: **Single-image dehazing via the Dark Channel Prior**
Non-medical restoration; the only Idea-Bank topic where the classical-vs-learned question stays *genuinely open*;
viva-defensible classical operators; paired-GT benchmarks (RESIDE/O-HAZE) → quantitative PSNR/SSIM. Full plan:
**[Research Project/RESEARCH_PLAN.md](Research Project/RESEARCH_PLAN.md)**. (Runner-up: cell-counting via watershed.)
