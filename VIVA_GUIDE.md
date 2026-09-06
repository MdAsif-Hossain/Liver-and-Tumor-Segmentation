# VIVA_GUIDE.md — Building a bilingual viva-prep document from your project

A reusable playbook for turning a finished project into a study document you can revise from and print
to PDF. Written from what worked while preparing the CSE 438 Part-B viva.

Third in the set: `GUIDE.md` (code) → `REPORT_GUIDE.md` (report) → **this** (viva).

---

## The principle that shapes everything

> **Build a study guide, not a script.**

A scripted answer survives exactly one question. The examiner's *second* question — "why?", "what if
you changed X?", "show me that in the code" — is where memorised text collapses and understanding
shows. So the document's job is to make you able to **re-derive** each answer, not recite it.

Practical consequence: every entry explains the *mechanism*, not just the fact. "τ sharpens the
similarity distribution" is recitable. "Small τ punishes hard negatives more, which is why it controls
how tightly the embedding clusters" is understandable — and it answers the follow-up too.

This also keeps you clear of the AI-use rules most briefs carry (they typically forbid AI-generated
*viva answers* while permitting help understanding concepts).

---

## Why HTML → print → PDF, not LaTeX

If your document is monolingual English, LaTeX is fine. For anything with **Bangla, Hindi, Arabic,
CJK** etc.:

| | LaTeX | HTML → browser print |
|---|---|---|
| Non-Latin script | needs XeLaTeX + `polyglossia` + a font that must be installed | browsers render it natively |
| Font fallback | fails hard if font missing | falls back down a stack |
| Setup time | 30+ min of debugging | zero |
| Print to PDF | native | Ctrl+P → Save as PDF |

Browsers already solve complex-script shaping. Don't refight that battle.

---

## The four-part structure (use this order)

Order matters — each part is the foundation for the next.

### 1 · Basic terms — *what / why / what it does*
Before any question, define the vocabulary. Use the same three-beat pattern every time:

> **What:** one sentence definition.
> **Why:** why the field needs it / what problem it solves.
> **Does:** what it concretely produces or changes.

That triad is what an examiner is actually testing when they ask "what is X?" — a definition alone
sounds memorised; definition + motivation + effect sounds understood.

Pick terms by scanning your own report and notebooks for **every noun you'd be embarrassed not to
define**. Ours: segmentation, SSL, pretext task, encoder/decoder, frozen vs fine-tuned, label
efficiency, augmentation/views, EMA, representation collapse, patch tokens, ASPP, 2.5D input, class
imbalance, leakage-safe split.

### 2 · Equations
For each one: the formula, then **a plain-language reading of what each part does**.

```
NT-Xent:  ℓ = − log [ exp(sim(zi,zj)/τ) ÷ Σ_{k≠i} exp(sim(zi,zk)/τ) ]

Numerator = the positive pair. Denominator = all negatives.
Minimising pulls positives together and pushes negatives apart.
τ sharpens the distribution. Negatives are what prevent collapse.
```

The reading is the part you'll actually be asked about. A formula you can't narrate is a liability.

**Include:** every loss you used, every metric you report, and any derived ratio you quote. If a number
appears in your report, its formula belongs here.

### 3 · Results — what won, and *why*
A compact results table, then one short subsection per notable finding. For each: **the observation,
then the mechanism.**

Cover, at minimum:
- the best method — and *why* its objective suited this data
- the most surprising/interesting result — usually where the real marks are
- the worst method — and your hypothesis for it
- any place your result contradicts expectation

### 4 · The actual viva questions
Split them into two kinds, because they need opposite treatment:

| Question type | Treatment |
|---|---|
| **About your code/your run** ("show me your masking code", "walk through your experiment") | Give an **anchor**: which cell/function to open + the 2–3 key facts. The answer must come from *your* notebook — you'll be asked to point at it. |
| **Conceptual/theory** ("why doesn't BYOL collapse?") | Full explanation — these are standard concepts with textbook answers; learn them properly. |

Writing scripted first-person prose for the first kind is the classic mistake: you can recite it but
you can't find the code, and that's immediately visible.

---

## Bilingual layout

Pattern: **English paragraph, then a visually distinct translated block.** Not interleaved
sentence-by-sentence — that breaks reading flow in both languages.

```html
<p>English explanation here.</p>
<div class="bn">বাংলা ব্যাখ্যা এখানে।</div>
```

**Translation guidance:** keep technical terms in English inside the translated text
(`encoder`, `loss`, `pretext task`). That is how the material is actually taught and discussed, and
inventing native-language equivalents for standard jargon makes it *harder* to read, not easier. The
translation should carry the *explanation*, not re-coin the vocabulary.

Keep the translated block **shorter than the English** — it's a reinforcement pass, not a duplicate.

---

## Reusable HTML skeleton

Copy this and fill in. It is print-ready for A4.

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Viva Preparation — COURSE CODE</title>
<style>
  @page { size: A4; margin: 16mm 14mm; }
  :root{ --ink:#111; --ink2:#444; --muted:#777; --rule:#d8d8d2;
         --accent:#2a78d6; --warn:#eda100; --bg-soft:#f7f7f4; }
  body{ font-family:"Segoe UI",system-ui,sans-serif; color:var(--ink);
        line-height:1.5; font-size:10.6pt; max-width:190mm; margin:0 auto; padding:8mm 4mm; }

  /* --- non-Latin block: font stack matters most --- */
  .bn{ font-family:"Nirmala UI","Noto Sans Bengali","Noto Serif Bengali","Vrinda",sans-serif;
       color:#1a3a5c; line-height:1.75; font-size:10.4pt;
       background:#f4f8fc; border-left:3px solid var(--accent);
       padding:6px 10px; margin:5px 0 0; border-radius:0 3px 3px 0; }

  h2{ font-size:13.5pt; margin:22px 0 8px; padding:6px 10px;
      background:var(--ink); color:#fff; border-radius:3px; }
  h3{ font-size:11.4pt; margin:14px 0 4px; border-bottom:1.5px solid var(--rule);
      padding-bottom:3px; }
  .q{ font-weight:700; margin:14px 0 4px; padding-left:9px;
      border-left:3px solid #eb6834; }              /* question marker */
  .eq{ background:var(--bg-soft); border:1px solid var(--rule); border-radius:4px;
       padding:8px 12px; margin:7px 0; text-align:center;
       font-family:"Cambria Math",Cambria,Georgia,serif; font-size:11pt; }
  .note{ background:#fff8e6; border-left:3px solid var(--warn);
         padding:7px 10px; margin:8px 0; border-radius:0 3px 3px 0; }
  .term{ font-weight:700; color:var(--accent); }
  .pill{ display:inline-block; background:var(--bg-soft); border:1px solid var(--rule);
         border-radius:10px; padding:1px 8px; font-size:9pt; }
  table{ border-collapse:collapse; width:100%; margin:8px 0; font-size:9.8pt; }
  th{ background:#eceae5; text-align:right; padding:5px 8px;
      border-bottom:1.5px solid var(--rule); }
  th:first-child, td:first-child{ text-align:left; }
  td{ padding:4px 8px; border-bottom:1px solid #eee; font-variant-numeric:tabular-nums; }
  .best{ background:#dff0e4!important; font-weight:700; }
  .avoid-break{ break-inside:avoid; }              /* keeps a Q+A on one page */
</style></head>
<body>
  <h1>Viva Preparation — COURSE</h1>
  <h2>1 · Basic Terms</h2>
  <h3 class="avoid-break">1.1 Term</h3>
  <p><b>What:</b> … <b>Why:</b> … <b>Does:</b> …</p>
  <div class="bn">…</div>

  <h2>2 · Equations</h2>
  <div class="eq">formula</div>
  <p>plain-language reading</p>

  <h2>3 · Results</h2>
  <table>…</table>

  <h2>4 · Viva Questions</h2>
  <div class="q">Q1. …</div>
  <p><b>Show:</b> which cell. <b>Key facts:</b> …</p>
</body></html>
```

**The two CSS rules that matter most:**
- `.avoid-break{break-inside:avoid}` on each `<h3>` block — stops a question splitting across pages
- the **font stack** on `.bn` — list several fonts; on Windows `Nirmala UI` ships with the OS, so it
  works with no installation

**Export:** open in browser → Ctrl+P → Destination "Save as PDF" → enable *Background graphics*
(otherwise the coloured blocks print white).

---

## Choosing what goes in

Work backwards from the brief. Most briefs list expected viva questions — those are the spine. Then:

1. **Copy the brief's question list verbatim** into part 4. Answer every one.
2. **Extract every term** those questions rely on → part 1.
3. **Extract every formula** those terms rely on → part 2.
4. **Add anything in your report you'd struggle to defend** — every claim in your results and insights
   sections is fair game, so anything you can't explain needs an entry.

If the brief has no question list, generate one by asking of each report section: *"what would I ask
someone who wrote this?"*

---

## Build in your defensive points

The strongest viva move is **volunteering your own limitations before the examiner finds them**. It
converts an attack into evidence of rigour. Give these their own highlighted box.

Ours were:
- the headline number came from a checkpoint selected on the test set → always quote the unbiased
  figure alongside it
- the best method differed in four ways at once → its win isn't cleanly attributable to one cause
- one finding was "consistent with" a hypothesis, not proof → and we named the diagnostic that would
  settle it

For your project, list every place where: a result is confounded, a metric is biased, an n is small,
or a conclusion outruns the evidence. **Each one, plus what you'd do about it, is a prepared answer.**

---

## Final page: the cheat-sheet

End with 3–5 numbers you must have instantly available — the ones that appear in your headline claims.
Ours: `19% of labels` · `frozen→fine-tuned +0.0739` · `MAE best liver 0.882 / worst tumour 0.287`, plus
the pairing rule `"101%" always with "97.6% unbiased"`.

If you can't recall these without looking, you're not ready.

---

## Checklist

```
[ ] Decide: study guide, not script — every entry explains a mechanism
[ ] Part 1: terms, each as what / why / does
[ ] Part 2: every loss, metric and ratio you quote — formula + plain reading
[ ] Part 3: results table + why each notable finding happened
[ ] Part 4: brief's questions verbatim; code Qs get anchors, theory Qs get explanations
[ ] Translated block after each English block; jargon stays in English
[ ] avoid-break on question blocks; font stack with an OS-bundled fallback
[ ] Defensive-points box: confounds, biases, small n, over-claims
[ ] Cheat-sheet of headline numbers on the last page
[ ] Print to PDF with "Background graphics" ON; check the non-Latin text rendered
```
