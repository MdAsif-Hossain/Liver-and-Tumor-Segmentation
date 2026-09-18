# Research Project — Single-Image Dehazing via the Dark Channel Prior

**Course:** CSE 348 / 438 — Digital Image Processing · **Group 01**, Dept. of CSE, East West University
**Members:** Md. Asif Hossain (2022-3-60-007) · Nabil Subhan (2022-3-60-063) · K M Nudar (2022-3-60-234)
**Plan:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md) · **Report:** [report/report.tex](report/report.tex) (IEEEtran, numbers pulled from the run)

> **Research question.** Which stage of the Dark Channel Prior pipeline governs restoration quality, how does each stage's
> contribution shift with haze density, and where does DCP fail — and how does a tuned, fully classical DCP compare with a
> learned dehazer on the *same* images?
>
> **H1** refinement dominates quality · **H2** atmospheric-light errors drive bright/sky-region failure · **H3** DCP's
> advantage shrinks with haze density.

Everything runs on **Kaggle** (CPU is enough for NB 1–2; a GPU only speeds up NB 3). Nothing is downloaded at run time.

---

## 1. Notebooks (run in this order)

| # | Notebook | What it does | Key outputs (`/kaggle/working`) |
|---|---|---|---|
| 1 | `dehaze-dcp-pipeline.ipynb` | From-scratch DCP (dark channel → *A* → raw *t* → refinement → recovery) with 4 *A*-estimators and 3 refinement operators (none / guided filter / matting Laplacian). Data inventory, scene-grouped tune/test split with fingerprint, stage visualisation, density-measure validation, sanity vs the hazy input. | `tab_dataset_inventory`, `fig_pipeline_stages`, `fig_refinement_compare`, `fig_density_validation`, `fig_sanity_grid`, `dehaze_records.json`, `dehaze_image_stats.csv` |
| 2 | `dehaze-ablation-eval.ipynb` | **The research notebook.** One-factor-at-a-time ablation (19 configurations × every image) with paired bootstrap CIs and Wilcoxon tests; stage-importance ranking (H1); matting-vs-guided subset; tuned configuration *selected on tune, reported on test*; density strata + refinement×density cross-cut (H3); bright-region failure analysis per *A*-estimator (H2); best/worst grids. | `dehaze_per_image_scores.csv`, `dehaze_ablation_summary.csv`, `tab_ablation_*`, `tab_stage_importance`, `tab_methods_test`, `tab_density`, `tab_failure_bright`, `fig_ablation_*`, `fig_density`, `fig_failure_bright`, `fig_best_worst`, **`dehaze_results.json`**, **`numbers.tex`** |
| 3 | `dehaze-learned-reference.ipynb` | Learned reference **AOD-Net** (pretrained weights if attached, else trained here — on RESIDE-ITS/OTS if attached, else on synthetic haze rendered from *tune-split* ground truths only) + CLAHE control, evaluated on the identical test split and metrics; learned-vs-classical gap by density stratum. | `tab_final_comparison`, `tab_final_by_dataset`, `tab_final_by_stratum`, `fig_final_comparison`, `fig_learned_gap`, `fig_qualitative_methods`, `dehaze_learned_results.json`, **`numbers_learned.tex`** |

Every figure is written as PNG (for notebooks) **and** PDF (for the report); every table as CSV **and** PNG.

The first code cell of each notebook is the **same shared library** (DCP operators, metrics, data discovery, statistics,
figure style). If you change it in one notebook, copy it to the other two.

## 2. Kaggle setup

**Accelerator:** *None* for NB 1 and NB 2 (pure NumPy/OpenCV; NB 2 takes roughly 1.5–2 h on ~1,100 images).
For NB 3 choose *GPU T4* if you train AOD-Net (minutes); CPU works too (slower). **Internet:** not needed.

**Attach (Add Data → search):** any subset of

| Dataset | Kaggle search terms | Layout the notebooks expect | Role |
|---|---|---|---|
| RESIDE **SOTS** indoor + outdoor | "RESIDE SOTS", "reside standard" | `…/indoor/hazy/1400_1.png` ↔ `…/indoor/gt/1400.png`; `…/outdoor/hazy/0001_0.8_0.2.jpg` ↔ `…/outdoor/gt/0001.png` | primary synthetic benchmark (perfect pairs) |
| **O-HAZE** (NTIRE 2018) | "O-HAZE" | `hazy/01_outdoor_hazy.jpg` ↔ `GT/01_outdoor_GT.jpg` | real outdoor haze |
| **I-HAZE** (NTIRE 2018) | "I-HAZE" | `hazy/01_indoor_hazy.jpg` ↔ `GT/01_indoor_GT.jpg` | real indoor haze |
| **Dense-Haze** (NTIRE 2019) / **NH-HAZE** (2020) | "Dense-Haze", "NH-HAZE" | `hazy/01_hazy.png` ↔ `GT/01_GT.png` | dense / non-homogeneous real haze (H3 hard stratum) |
| RESIDE **ITS / OTS** (optional) | "RESIDE ITS" | `ITS/hazy/1_1_0.90.png` ↔ `ITS/clear/1.png` | **training only** — auto-excluded from evaluation, used by NB 3 to train AOD-Net |
| AOD-Net weights (optional) | upload `dehazer.pth` / any AOD-Net `*.pth` as a private dataset | five conv layers | NB 3 loads it instead of training |

Discovery rules (see `discover_pairs` in the library cell): a folder named `hazy`/`haze`/`input` next to a folder named
`GT`/`gt`/`clear`/`clean`/`target`, paired by the **leading number** of the file name; datasets are labelled from their
path (`sots`+`indoor`/`outdoor`, `o-haz`, `i-haz`, `dense`, `nh`); anything under `its`/`ots`/`train` is training-only.
The inventory table printed by NB 1 tells you exactly what was found — check it before running NB 2.

**Output chaining:** after NB 2 finishes, *Save Version* and attach **NB 2's output** to NB 3 (it reads
`dehaze_results.json` for the tuned configuration and checks the split fingerprint).

**Caps and speed:** `CONFIG["max_per_dataset"]` (default 500) subsamples large sets with a fixed seed;
`CONFIG["max_side"]` (512) sets the working resolution; set the environment variable `DEHAZE_FAST=1` for a smoke run.

**With nothing attached** every notebook still runs on a synthetic-haze fallback (`I = J·t + A(1−t)` on public test images)
so the code can be checked anywhere — those numbers are labelled `synthetic` and must not be reported as benchmark results.

## 3. Evaluation protocol (what makes it a study, not a demo)

* **Scene-grouped tune/test split** (30 % / 70 % of scenes per dataset, never more than half the scenes in tune), seeded and
  fingerprinted; all hazy renderings of one SOTS scene stay on the same side.
* **Selection on tune, reporting on test.** Every configuration is scored on every image; the tuned DCP is chosen on tune
  and every table in the report is the test split.
* **Paired statistics.** Mean per-image difference, 95 % bootstrap CI (1,000 resamples), Wilcoxon signed-rank *p*;
  Spearman ρ and Kruskal–Wallis for density effects.
* **Metrics.** PSNR, SSIM, CIEDE2000 (full-reference); Hautière *e* and *r̄* (no-reference); mean dark channel of the
  input as a no-reference haze-density measure (validated in NB 1), tercile strata light / medium / dense.
* **Bright-region masks** are computed on the *clear* image so the H2 analysis is not confounded by haze density.
* **Learned reference** on the identical test images, with the weight-provenance route recorded.

## 4. From the Kaggle outputs to the report

1. Download the outputs of NB 1–3 (or attach them to a scratch notebook and `!cp`).
2. Copy all `fig_*.pdf` and `tab_*.png` into `report/figures/`, and `numbers.tex` + `numbers_learned.tex` next to
   `report/report.tex`.
3. Compile `report.tex` with pdfLaTeX (Overleaf works). Any `??` in the PDF is a macro the run did not produce.
4. Write the **Discussion and Insights** section yourselves (prompts are in the `.tex`), then run the verification pass
   from [../REPORT_GUIDE.md](../REPORT_GUIDE.md) — every number in the prose must trace to a table.

## 5. Files

```
Research Project/
├── README.md                      this file
├── RESEARCH_PLAN.md               research question, hypotheses, ablation matrix, risks
├── dehaze-dcp-pipeline.ipynb      NB 1
├── dehaze-ablation-eval.ipynb     NB 2
├── dehaze-learned-reference.ipynb NB 3
└── report/
    ├── report.tex                 IEEEtran, reads numbers.tex / numbers_learned.tex, figures/
    └── figures/                   <- copy fig_*.pdf and tab_*.png from the Kaggle outputs here
```
