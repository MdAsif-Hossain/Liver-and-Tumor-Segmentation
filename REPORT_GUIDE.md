# REPORT_GUIDE.md — Turning notebook outputs into a submitted report

A reusable playbook, written from what actually worked (and what broke) while producing the CSE 438
Part-B report. Domain-agnostic: works for any assignment/paper where results live in notebooks and the
deliverable is a LaTeX PDF.

Companion to `GUIDE.md` (which covers the *code* side). This one starts the moment your runs finish.

---

## The one rule that matters

> **Every number in the report must be traceable to a printed output. Verify each one before you
> submit — not by trusting your memory of the run, but by re-reading the output.**

In our report this rule caught a real error (`47%` where the data said `73%`) that had already survived a
full compile and a read-through. It costs 20 minutes and it is the difference between a report that
holds up under questioning and one that doesn't.

---

## Phase 0 — Freeze before you write

1. **Stop changing the models.** Once you start writing, results are frozen. If you retrain, every
   number, table and figure is stale and you will miss one.
2. **Collect all outputs in one folder.** Copy the executed `.ipynb` files (with outputs) into an
   `outputs/` directory. The report is written *from these files*, never from memory.
3. **Make one results table by hand first** — every metric, every method, in a scratch file. If you
   can't fill a cell, you have a gap in your experiments, not in your writing.
4. **Put the report folder under version control** (`git add`) *before* you start editing.
   We didn't, and when a bad edit destroyed three tables there was no rollback.

---

## Phase 1 — Extract figures programmatically

Never screenshot. Pull the PNGs straight out of the notebook JSON so they are exactly what the code
produced.

```python
import json, base64
from pathlib import Path
out = Path("report/figures"); out.mkdir(parents=True, exist_ok=True)
nb = json.load(open("outputs/my-notebook.ipynb", encoding="utf-8"))

# 1) FIRST: print an inventory so you know what is where
for ci, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    imgs = [o for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    if not imgs: continue
    saves = [l.strip() for l in "".join(c["source"]).splitlines() if "savefig" in l]
    print(f"cell {ci}: {len(imgs)} image(s)")
    for s in saves: print("     ", s[:90])

# 2) THEN: map by (cell index, image index) — read off the inventory above
MAP = {(6, 0): "results_bars.png", (6, 1): "per_class_iou.png"}
for (ci, k), fn in MAP.items():
    imgs = [o["data"]["image/png"] for o in nb["cells"][ci].get("outputs", [])
            if "image/png" in o.get("data", {})]
    (out / fn).write_bytes(base64.b64decode(imgs[k]))
```

> ### ⚠️ Pitfall that bit us
> Our first version matched cells by **keyword** (`if "per_class_iou" in cell_source`) and then took
> `imgs[0]`. But one cell plotted **two** figures, so two different filenames both received the *first*
> image — and the compiled PDF showed a radar chart under a caption saying "Per-class IoU". It survived
> a full read-through because captions look right in the source.
>
> **Always map by `(cell index, image index)`, and always open the extracted PNGs and look at them.**

---

## Phase 2 — Build the skeleton from the brief, not from your head

Open the assignment brief's report-structure section and create **one LaTeX section per required
subsection, in their order, with their names**. Put a `% TODO` in each.

This does two things: it guarantees you can't omit a required section, and it makes the graders' job
mechanical — they can tick their own list against your headings.

```latex
\section{Introduction}          % 6.2
\section{Dataset and Split}     % 6.3
\section{Methodology}           % 6.4
...
```

Then, before writing a word, run a coverage check:

```python
import re
need = ["Introduction", "Dataset", "Methodology", "Results", "Error Analysis", "Conclusion"]
secs = re.findall(r"\\section\{([^}]+)\}", open("report.tex", encoding="utf-8").read())
print("missing:", [n for n in need if not any(n.lower() in s.lower() for s in secs)])
```

---

## Phase 3 — Write the factual sections first

Order matters. Write in this sequence, because each one is easier once the previous is done:

| Order | Section | Source of truth |
|---|---|---|
| 1 | **Results table** | your scratch table from Phase 0 |
| 2 | Dataset & Split | data-prep notebook printouts (exact counts!) |
| 3 | Methodology | your config cells — quote the actual hyperparameters |
| 4 | Setup / training | training logs (per-epoch times, epoch counts) |
| 5 | Error analysis *evidence* | per-image CSVs, confusion matrices |
| 6 | Abstract | last — it summarises what you now know |
| 7 | Conclusion | last |

**Rules while writing:**
- Quote numbers to the precision your output printed. Don't round `0.7757` to `0.78` in one place and
  `0.776` in another.
- Every figure and table must be **cross-referenced in the body text** (`Fig.~\ref{...}`). A float
  nobody mentions looks like padding.
- **Don't restate a caption in the body.** We wasted ~90 words duplicating captions verbatim; cutting
  them lost no information and helped the page count.

---

## Phase 4 — The verification pass (do not skip)

This is the highest-value hour of the whole process.

### 4a. Every numerical claim
Make a list of every number in the prose, then find its source in an output cell. Anything you can't
trace gets deleted or corrected.

```python
import re
tex = open("report.tex", encoding="utf-8").read()
body = re.sub(r"(?m)^\s*%.*$", "", tex)                    # strip comments
nums = sorted(set(re.findall(r"\d+\.\d{2,}|\d{1,3}(?:,\d{3})+|\b\d{2,}\%", body)))
print(len(nums), "numeric claims to verify:"); print(nums)
```

### 4b. Arithmetic you asserted
Re-compute every derived figure. Ours:

```python
print("ratio      :", 2553/13447)            # claimed 0.190 ✓
print("x more data:", 13447/2553)            # claimed 5.3x  ✓
print("% no-tumour:", (3158-844)/3158)       # claimed 47% ✗ — actually 73%
```

That last line is a real error we shipped into a compiled PDF and only caught here.

### 4c. Figures match their captions
Open each extracted PNG and confirm it shows what the caption says. **Two of ours didn't.**

### 4d. Structural validation (run before every compile)

```python
import re
from pathlib import Path
tex = Path("report.tex").read_text(encoding="utf-8")
figs = {p.name for p in Path("figures").glob("*.png")}

inc = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
print("missing figures:", [f for f in set(inc) if f not in figs] or "none")
print("braces:", tex.count("{"), "/", tex.count("}"))
for env in ["document","figure","table","table\\*","tabular","abstract","thebibliography"]:
    o = len(re.findall(r"\\begin\{"+env+r"\}", tex)); c = len(re.findall(r"\\end\{"+env+r"\}", tex))
    if o != c: print(f"  UNBALANCED {env}: {o} begin / {c} end")
labels = set(re.findall(r"\\label\{([^}]+)\}", tex)); refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
print("undefined refs:", sorted(refs-labels) or "none")
print("unused labels :", sorted(labels-refs) or "none")
print("broken refs   :", len(re.findall(r"ef\{", tex)) - len(re.findall(r"\\ref\{", tex)))
```

The last line catches `\ref` that lost its backslash — which renders as literal text like
`Fig. effig:qual` in the PDF. We shipped exactly that.

---

## Phase 5 — Making it fit the page limit

Apply in this order — cheapest (loses nothing) to most expensive (loses content).

**1. Float spacing — the biggest free win.** LaTeX leaves ~12pt above *and* below every float. With 15
floats that is over two inches of white space.

```latex
\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{0pt}
```

**2. Shrink figure widths.** Tall multi-panel image grids are the page hogs — a 4×4 grid at
`\columnwidth` eats a third of a column. Drop those to `0.5–0.6\columnwidth` first; wide short charts
barely matter.

**3. Compress the header.** Ours went from 5 lines to 3 by putting student IDs inline with names
instead of on a separate footnote-marker line.

**4. Cut caption-duplicating prose** (see Phase 3).

**5. Tables to `\footnotesize`.**

**6. Only now: remove figures.** Cut *extras* first, never anything the brief requires. Rank yours as
required / supporting / decorative and delete from the bottom. **Remove the `\ref` to it too** — an
orphan label is a compile warning and a dangling reference is a visible bug.

### Table wider than the column
Symptom: numbers from the table overprint the neighbouring column's text. Fix:

```latex
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrr}
 ... 
\end{tabular}}
```

Also shorten the cell text — `\resizebox` shrinks the font to fit, so a very wide table becomes
unreadably small.

---

## Phase 6 — Final compliance audit

Re-read the brief line by line and tick each requirement against the compiled PDF (not the source):

- [ ] Every required section present, correctly named
- [ ] Word/page limits (**count the compiled PDF**, not your estimate)
- [ ] Abstract within its word range
- [ ] Required table(s) and chart(s) present
- [ ] Figures referenced in text; captions match content
- [ ] References cited **in the body**, not just listed (a bibliography with zero `\cite` is an obvious tell)
- [ ] Author names, IDs, group, course, dataset — all in the header
- [ ] Any mandated disclosure actually written (limitations, caveats, protocol departures)

---

## Pitfalls table — every one of these cost us real time

| Pitfall | Symptom | Prevention |
|---|---|---|
| **Regex across LaTeX environments** | A caption-shortening regex spanned from a table's `\caption{}` to the next `\label{fig:}` and **deleted three tables** | Never regex across environments. Use exact literal `.replace()` and `assert text.count(old) == 1` |
| Figure extraction by keyword | Wrong image under a correct caption | Map by `(cell, image index)`; *look at* every extracted PNG |
| Bash heredoc eats backslashes | `\ref{x}` becomes `ef{x}`, prints as literal text | Write patch scripts to a **file** and run the file; never inline LaTeX-editing regex in a heredoc |
| Report not in git | No rollback after a destructive edit | `git add` the report folder before the first edit |
| Rounding drift | `0.78` here, `0.776` there | Fix precision once and grep for each number |
| Unverified derived stats | Shipped `47%` where data said `73%` | Phase 4b — recompute everything |
| Caption duplicated in body | Wasted space, reads padded | Caption states *what it is*; body states *what it means* |
| Mixed metric conventions | Quoting a selected-checkpoint number beside an unbiased one without saying which | Label the convention every time you quote a number |

---

## Writing quality: three habits that raise the grade

**1. Quantify your own limitations before a reader finds them.** Our report reported a `101% of
baseline` headline *and* the `97.6%` unbiased figure, and explained why the gap exists. Naming your
own bias reads as rigour; hiding it reads as either sloppiness or spin — and it is the first thing an
examiner probes.

**2. Say what a number *means*, not just what it is.** "MAE achieved 0.8821 liver IoU and 0.2873
tumour IoU" is data entry. "MAE is best on the large smooth class and worst on the small one, because
pixel reconstruction rewards low-frequency structure" is analysis. Grades and viva marks live in the
second sentence.

**3. Name your confounds.** If two conditions differ in more than one way, say so explicitly. Ours:
DINOv2 differed in architecture, resolution, adapter *and* pretraining corpus — so "self-distillation
is best" was not a clean conclusion. Writing that down is worth more than pretending it's clean.

---

## If part of the report must be your own work

Many briefs forbid AI-generated analysis/insight/viva answers while permitting help with boilerplate.
The split that works:

- **Get help with:** structure, LaTeX mechanics, factual sections, tables, verification scripts,
  and *explanations of concepts* so you understand them.
- **Write yourself:** interpretation, insights, error-analysis reasoning, viva answers.
- **A good workflow:** have the evidence laid out for you (numbers, figures, what each shows), write
  the interpretation in your own words — even rough — then ask for a critique of the reasoning rather
  than a rewrite. You end up with a stronger section than you'd get handed, and you can defend it.

---

## Quick-start checklist for the next project

```
[ ] Freeze results; copy executed notebooks into outputs/
[ ] git add the report folder
[ ] Build the scratch results table by hand
[ ] Inventory notebook images; extract by (cell, index); LOOK at each one
[ ] Create one LaTeX section per required subsection, in the brief's order
[ ] Write: results → dataset → methodology → setup → evidence → abstract → conclusion
[ ] Verification pass: every number traced, all arithmetic recomputed, captions vs figures
[ ] Run the structural validator; compile; count pages
[ ] Fit to page limit: float spacing → figure widths → header → prose → cut extras
[ ] Final compliance audit against the brief, using the compiled PDF
```
