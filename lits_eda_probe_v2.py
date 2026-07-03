# =====================================================================
# LiTS 256 — EDA PROBE v2  (CORRECTED for the real 3-folder layout)
# Images/ + Liver_mask/ + Tumor_mask/  -> combine into 3-class label {0,1,2}
# Run as ONE Kaggle cell (CPU is fine). Then send Claude the SUMMARY + the
# eda_v2_samples.png figure.
# =====================================================================
import re, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

SEED = 42; random.seed(SEED); np.random.seed(SEED)

# ---- locate the 3 folders robustly ----
ROOT = Path("/kaggle/input/datasets/sodko3/lits-dataset-liver-and-tumor-segmentation-256x256/Thesis_data")
if not ROOT.exists():
    cands = list(Path("/kaggle/input").rglob("Thesis_data"))
    ROOT = cands[0] if cands else Path("/kaggle/input")
IMG_DIR   = next(p for p in ROOT.iterdir() if p.is_dir() and "image" in p.name.lower())
LIVER_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and "liver" in p.name.lower())
TUMOR_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and "tumor" in p.name.lower())
print("folders:", IMG_DIR.name, "|", LIVER_DIR.name, "|", TUMOR_DIR.name)

def key(p):  return tuple(int(x) for x in re.findall(r"\d+", p.stem))   # (volume, slice)
imgs = {key(p): p for p in IMG_DIR.glob("*.png")}
livs = {key(p): p for p in LIVER_DIR.glob("*.png")}
tums = {key(p): p for p in TUMOR_DIR.glob("*.png")}
keys = sorted(set(imgs) & set(livs) & set(tums))
print(f"triplets (image+liver+tumor): {len(keys)}")

def arr(p):
    a = np.array(Image.open(p)); return a[..., 0] if a.ndim == 3 else a

def combined_label(k):
    liv = arr(livs[k]) > 0; tum = arr(tums[k]) > 0
    lab = np.zeros(liv.shape, np.uint8); lab[liv] = 1; lab[tum] = 2   # tumor precedence
    return lab

# ---- TRUE 3-class pixel distribution + slice-level prevalence (sample) ----
N = min(6000, len(keys))
sample = random.sample(keys, N)
cls = np.zeros(3, np.int64)
has_liver = has_tumor = bg_only = 0
for k in sample:
    c = np.bincount(combined_label(k).ravel(), minlength=3)[:3]
    cls += c
    has_liver += int(c[1] > 0)
    has_tumor += int(c[2] > 0)
    bg_only   += int(c[1] == 0 and c[2] == 0)
tot = cls.sum()

print("\n--- TRUE PIXEL-LEVEL CLASS DISTRIBUTION (combined 3-class) ---")
df = pd.DataFrame({"class": ["background", "liver", "tumor"], "pixels": cls,
                   "percent": np.round(100 * cls / tot, 4)})
print(df.to_string(index=False))

print(f"\n--- SLICE PREVALENCE over {N} sampled slices ---")
print(f"liver-present    : {has_liver:5d}  ({100*has_liver/N:5.1f}%)")
print(f"tumor-present    : {has_tumor:5d}  ({100*has_tumor/N:5.1f}%)")
print(f"background-only   : {bg_only:5d}  ({100*bg_only/N:5.1f}%)")
print(f"=> estimated liver-containing slices in FULL set: ~{int(len(keys)*has_liver/N)}")

freq = cls / tot
w = np.round(np.median(freq) / freq, 3)
print(f"\nmedian-frequency class weights [bg, liver, tumor] = {w.tolist()}")
print(f"imbalance ratio bg:liver:tumor = "
      f"{int(cls[0]/cls[2])} : {round(cls[1]/cls[2],1)} : 1  (per tumor pixel)")

# ---- CORRECTED 10-sample overlay with the COMBINED mask (prefer tumor slices) ----
tum_keys = [k for k in random.sample(keys, min(len(keys), 400)) if arr(tums[k]).max() > 0]
chosen = (tum_keys[:6] + random.sample(keys, 10))[:10]
cmap = plt.matplotlib.colors.ListedColormap(["black", "gold", "red"])  # bg / liver / tumor
fig, ax = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)))
for i, k in enumerate(chosen):
    im = arr(imgs[k]); lab = combined_label(k)
    ax[i, 0].imshow(im, cmap="gray"); ax[i, 0].set_title(f"vol-{k[0]} slice-{k[1]}", fontsize=7)
    ax[i, 1].imshow(lab, cmap=cmap, vmin=0, vmax=2)
    ax[i, 1].set_title(f"label uniq={sorted(np.unique(lab).tolist())}", fontsize=7)
    ax[i, 2].imshow(im, cmap="gray"); ax[i, 2].imshow(lab, cmap=cmap, vmin=0, vmax=2, alpha=0.45)
    ax[i, 2].set_title("overlay (liver=gold, tumor=red)", fontsize=7)
    for j in range(3): ax[i, j].axis("off")
plt.tight_layout(); plt.savefig("/kaggle/working/eda_v2_samples.png", dpi=110); plt.show()

# ---- SUMMARY ----
print("\n=================== SUMMARY v2 (paste to Claude) ===================")
print(f"triplets            : {len(keys)}   volumes: {len(set(k[0] for k in keys))}")
print(df.to_string(index=False))
print(f"liver slices ~{int(len(keys)*has_liver/N)}  tumor slices ~{int(len(keys)*has_tumor/N)}  "
      f"bg-only ~{100*bg_only/N:.1f}%")
print(f"class weights [bg,liver,tumor] = {w.tolist()}")
print("===================================================================")
