# Research Plan — Single-Image Dehazing via the Dark Channel Prior
## A stage-wise ablation study of a classical restoration pipeline, with a learned-method reference

**Course:** CSE 348 / 438 — Digital Image Processing · **Group 01**, Dept. of CSE, East West University
**Members:** Md. Asif Hossain (2022-3-60-007) · Nabil Subhan (2022-3-60-063) · K M Nudar (2022-3-60-234)
**Category (Idea Bank):** Filtering & Restoration → *Image dehazing using dark channel prior*
**Chosen by:** 3-advisor council synthesis (research-impact ✓ · non-medical ✓ · viva-defensible ✓ · feasible ✓)

> Swap note: to pivot to the runner-up (**cell counting via watershed**, CC0 microscopy data, max harness reuse),
> say so and I re-issue this plan. Everything below assumes dehazing.

---

## 1. Research question & hypotheses

**RQ:** *Which stage of the Dark Channel Prior (DCP) dehazing pipeline governs restoration quality, and how does
each stage's contribution shift with haze density — and how close does a fully classical, well-tuned DCP get to a
learned dehazer?*

- **H1 (refinement dominates quality):** transmission-map **refinement** (guided filter vs soft matting vs none)
  produces the largest PSNR/SSIM gains, exceeding the effect of patch size or ω.
- **H2 (atmospheric light dominates failure):** errors in **atmospheric-light `A`** estimation dominate the
  *failure rate* in bright/sky/white-object regions, largely independent of refinement.
- **H3 (density interaction):** DCP's advantage shrinks as haze density rises; on **dense** haze a learned model
  opens a clear gap, while on **light/medium** haze a tuned DCP is competitive.

These are *testable* — each maps to a measurable ablation, not a demo claim.

---

## 2. Why this is research, not a demo

DCP (He, Sun & Tang, CVPR 2009 best paper) decomposes into **independently ablatable stages**, each mapping to one
algorithmic operator. Unlike skin-lesion Otsu-vs-U-Net (a settled loss) or Hough lane detection (no standard
quantitative protocol), DCP remains **competitive/complementary** to deep methods, so the central comparison is an
open question. The team's evaluation-and-error-analysis discipline (from Parts A/B) is exactly what turns a
parameter sweep into a scientific result.

---

## 3. Datasets (paired ground truth → full-reference metrics)

| Dataset | Content | GT | License / access | Role |
|---|---|---|---|---|
| **RESIDE — SOTS** (Li et al. 2018) | 500 indoor + 500 outdoor synthetic hazy | ✅ clean pairs | research, Kaggle-mirrored | **primary benchmark** |
| **O-HAZE** (NTIRE'18) | 45 **real** outdoor hazy/clear pairs | ✅ | research | real-haze generalization |
| **I-HAZE** (NTIRE'18) | 35 **real** indoor pairs | ✅ | research | indoor real haze |
| **Dense-Haze** (NTIRE'19) | 55 **dense** real pairs | ✅ | research | H3 hard-case / density stratum |

Haze-density stratification (light / medium / dense) for H3 comes from SOTS β-levels + the O/I/Dense-Haze split.
Report exact image counts per stratum (the Part-A "report counts" discipline).

---

## 4. Method — the DCP pipeline (each stage a named classical operator)

1. **Dark channel:** `J_dark(x) = min_{c∈{r,g,b}} min_{y∈Ω(x)} J_c(y)` — per-pixel channel-min then local patch-min.
2. **Atmospheric light `A`:** from the top 0.1% brightest dark-channel pixels (baseline) — *ablated in §5*.
3. **Transmission estimate:** `t̃(x) = 1 − ω · darkchannel(I/A)`, ω≈0.95 (keeps slight haze for realism).
4. **Transmission refinement:** soft matting (original) **or** guided filter (He 2010) **or** none — *headline ablation*.
5. **Radiance recovery:** `J(x) = (I(x) − A)/max(t(x), t₀) + A`, floor `t₀≈0.1` to bound noise amplification.
6. **(Optional) post:** gamma / mild CLAHE on the recovered radiance — reported, not part of core claims.

Reference implementation in NumPy/OpenCV (fully classical, no learned weights); guided filter via `cv2.ximgproc`
or a from-scratch 20-line implementation (better for viva).

---

## 5. Ablation matrix — the research substance

| Stage | Levels swept | Tests |
|---|---|---|
| Dark-channel **patch size** Ω | 3, 7, 15, 31 px | halo/edge-artifact vs smoothness trade-off |
| **A estimation** | brightest-pixel · top-0.1% dark-channel · hierarchical quad-tree | **H2** (failure driver) |
| **ω** (haze retention) | 0.80, 0.90, 0.95, 0.99 | over/under-dehazing |
| **t₀** floor | 0.05, 0.10, 0.20 | noise amplification in dense haze |
| **Refinement** | none · guided filter (r, ε swept) · soft matting | **H1** (headline) |
| **Haze density** (strata) | light / medium / dense | **H3** interaction |

One-factor-at-a-time from a fixed baseline (the Part-A ablation methodology), plus the density cross-cut. This is
the core of the report and the viva.

---

## 6. Learned-method reference (contextual, not the main event)

Run 1–2 pretrained deep dehazers — **AOD-Net** (Li 2017, tiny) and/or **FFA-Net** (Qin 2020) — on the identical
test sets/metrics to position DCP. Framing: *complement/competitiveness*, not "classical loses." Kaggle-friendly
(inference only, pretrained). If weights are unavailable offline, DCP-vs-literature-numbers is an acceptable fallback.

---

## 7. Evaluation protocol & metrics

- **Full-reference:** PSNR, SSIM (primary); CIEDE2000 (color fidelity).
- **No-reference:** FADE (fog-aware density) and/or NIQE — for the real-haze sets where alignment is imperfect.
- **Per-image distributions + stratified means** (not just pooled) — reuse the Part-A per-image/per-patient
  reporting style: box plots per density stratum, best/worst grids.
- **Statistical care:** report mean ± std and paired differences across ablation levels on the same images.

*Harness reuse:* the multi-class IoU/ConfMat code doesn't transfer (this is restoration, not segmentation), but the
**per-image logging, stratified reporting, qualitative best/worst grids, and failure-analysis structure do** — that
discipline is the team's real transferable asset.

---

## 8. Failure analysis (the error-analysis edge)

Quantify DCP's documented breakdowns instead of just noting them:
- **Sky / bright regions:** DCP violates its own prior (dark channel not near 0) → color shift / over-darkening.
  Measure error *inside sky masks* vs elsewhere; tie to H2.
- **White / light objects:** transmission underestimated → artifacts.
- **Dense haze:** low transmission → noise amplification despite t₀. Tie to H3.
Present as a stratified error table + representative failure grid — the report's most original section.

---

## 9. Deliverables, compute & timeline

- **Notebook(s):** (1) `dehaze-dcp-pipeline` (implementation + stage visualizations + sanity), (2) `dehaze-ablation-eval`
  (the sweep + metrics + figures), optional (3) `dehaze-learned-reference`. Kaggle-public, outputs visible.
- **Report:** problem/RQ → method (per stage) → ablation results → density stratification → learned reference →
  failure analysis → insights → reproducibility → references. Figures: pipeline stages, ablation curves, PSNR/SSIM
  vs each knob, density box plots, best/worst grids, failure table.
- **Compute:** trivial — DCP is CPU/seconds per image; only the optional learned reference touches the GPU. No
  training. Fits Kaggle free tier comfortably.
- **Reproducibility:** fixed seeds, pinned versions, datasets cited with access notes, from-scratch guided filter.

---

## 10. Viva readiness (classical DIP mastery each member can defend)

Core operators every member must be able to derive/explain at the board: **local patch min & the dark-channel
statistic**, **atmospheric-light estimation**, the **haze imaging model** `I = J·t + A(1−t)` and inversion,
**transmission map**, and the **guided filter** (linear model, box-filter cost). These are canonical DIP —
whiteboard-able, no black boxes. Map each report claim → the cell that produces it (the Part-A "question→cell"
study checklist).

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Real-haze pairs slightly misaligned → PSNR unreliable | add no-reference FADE/NIQE; keep SOTS (perfectly paired) as primary |
| Guided-filter library (`ximgproc`) missing on Kaggle | ship a 20-line from-scratch guided filter (also better for viva) |
| Learned baseline weights unavailable offline | fall back to published-number comparison; DCP-ablation stands alone |
| Sweep explosion | one-factor-at-a-time from a fixed baseline + one density cross-cut, not full grid |
| "Just re-implementing DCP" critique | the contribution is the **stage-attribution + density-interaction + quantified failure analysis**, not the algorithm |

---

## 12. Key references

- He, Sun & Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 (best paper) / TPAMI 2011.
- He, Sun & Tang, *Guided Image Filtering*, ECCV 2010 / TPAMI 2013.
- Li et al., *Benchmarking Single-Image Dehazing and Beyond (RESIDE)*, IEEE TIP 2019.
- Ancuti et al., *O-HAZE / I-HAZE*, NTIRE 2018; *Dense-Haze*, NTIRE 2019.
- Li et al., *AOD-Net*, ICCV 2017 · Qin et al., *FFA-Net*, AAAI 2020 (learned references).
- Choi et al., *Referenceless Prediction of Perceptual Fog Density (FADE)*, IEEE TIP 2015.
