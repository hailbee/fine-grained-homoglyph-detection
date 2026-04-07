# Hit-Zone Aware Embedding Models for Fine-Grained Visual Homoglyph Detection

## Project structure

The pipeline runs in this order:

```
data/raw/  →  training/make_splits.py  →  data/splits/
                                                 ↓
                                    training/train.py  (or strip_design_sweep.py)
                                                 ↓
                                         outputs/runs/
                                                 ↓
                                    evaluation/summarize.py
```

---

### `rendering/`

| File | Purpose |
|------|---------|
| `renderer.py` | Renders a string to a fixed-height grayscale NumPy array using DejaVu Sans. Width scales with the string; height defaults to 32 px. Returns a `float32` array in `[0, 1]`. |
| `slicer.py` | Slices a rendered image into overlapping or non-overlapping column strips. Returns a `float32` array of shape `(num_slices, height, slice_width)` normalised to `[0, 1]`. Default: `slice_width=4`, `stride=4`. |

---

### `data/`

| File | Purpose |
|------|---------|
| `raw/domains_spoof.pkl` | Raw dataset of `(name_a, name_b, label)` pairs with train/validate/test keys. Source of truth — do not modify. |
| `raw/process_spoof.pkl` | Alternate processed version of the raw pairs. |
| `splits/train.pkl` | Training split produced by `make_splits.py`. |
| `splits/val.pkl` | Validation split. |
| `splits/test.pkl` | Test split. |

---

### `training/`

| File | Purpose |
|------|---------|
| `make_splits.py` | **Run first.** Reads `data/raw/domains_spoof.pkl` and writes the three split pkl files under `data/splits/`. Only needs to be run once. |
| `dataset.py` | `NamePairDataset` — renders and slices name pairs on-the-fly; `collate_fn` pads variable-length slice sequences into `(B, max_slices, height, slice_width)` tensors. |
| `train.py` | **Single training run.** Reads hyperparameters from `configs/default.yaml` (or `--config`). Saves `best.pt`, `config.yaml`, and `log.csv` to `outputs/runs/<run_name>/`. |
| `strip_design_sweep.py` | **Hyperparameter sweep.** Trains across all combinations of pooling, background, slice width, stride, and padding removal. Appends each result to `outputs/results.csv`. Supports `--resume` to continue an interrupted sweep. |

---

### `models/`

| File | Purpose |
|------|---------|
| `encoder.py` | `VisualEncoder` — takes `(batch, num_slices, height, slice_width)`, flattens each slice, runs two Conv1D layers, and pools into a `(batch, embed_dim)` embedding. Supports `mean`, `max`, and `attention` pooling. |
| `similarity.py` | `SimilarityHead` — computes cosine similarity between two embeddings, passes the scalar through a linear layer to produce a binary logit. Use with `BCEWithLogitsLoss`. |

---

### `configs/`

| File | Purpose |
|------|---------|
| `default.yaml` | Base config for all training runs. Controls rendering height, background colour, slice width/stride, model embed dim, pooling strategy, and training hyperparameters. Copy and modify to create experiment configs. |

---

### `evaluation/`

| File | Purpose |
|------|---------|
| `summarize.py` | **Run after the sweep.** Reads `outputs/results.csv` and prints the top-N configurations by val AUC plus the marginal effect of each design axis (pooling, background, slice width, padding). |

---

### `notebooks/`

| File | Purpose |
|------|---------|
| `explore.ipynb` | Visual sanity-checks for the rendering and slicing pipeline. Shows rendered images, per-slice 2-D heatmaps, a difference map between a genuine and spoofed pair, and the effect of varying `slice_width`/`stride` configs. |

---

### `outputs/`

| Path | Purpose |
|------|---------|
| `runs/<run_name>/best.pt` | Best model checkpoint (highest val AUC) for that run. |
| `runs/<run_name>/log.csv` | Epoch-level metrics: `train_loss`, `val_loss`, `val_auc`. |
| `runs/<run_name>/config.yaml` | Exact config used for that run. |
| `results.csv` | Aggregated sweep results — one row per sweep combination. |

---

## Running on an SSH / cluster machine

### 1. What to upload

Upload the project directory excluding `.venv/` and any `__pycache__/` folders:

```
fine-grained-homoglyph-detection/
├── configs/
├── data/
├── evaluation/
├── models/
├── notebooks/          # optional — only needed for exploration
├── rendering/
├── training/
├── outputs/            # can start empty; created automatically
└── requirements.txt
```

Copy to the cluster using `scp` (do not upload `.venv` — it will be built on the cluster):

```bash
scp -r fine-grained-homoglyph-detection/ <user>@<host>:~/
```

Or use a GUI SFTP client such as **WinSCP** (Windows) — drag and drop the folder, excluding `.venv/`.

---

### 2. Set up the environment (on the cluster)

```bash
cd ~/fine-grained-homoglyph-detection

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> If the cluster provides a PyTorch module (e.g. via `module load pytorch`), load it before creating the venv to avoid re-downloading large binaries.

---

### 3. Run order

**Step 1 — Make splits** (only needed once; skip if `data/splits/` already contains the pkl files)

```bash
python training/make_splits.py
```

**Step 2a — Single training run** (to verify everything works)

```bash
python training/train.py
# or with a custom config:
python training/train.py --config configs/my_experiment.yaml
```

Before running the sweep, open `configs/default.yaml` and set `num_workers` to a value greater than 0 (e.g. 4) — the default of 0 is safe on Windows but slow on Linux.

**Step 2b — Full hyperparameter sweep** (the main experiment)

```bash
python training/strip_design_sweep.py

# Resume an interrupted sweep without re-running finished combos:
python training/strip_design_sweep.py --resume

# Quick smoke test (5 epochs, 3 combos):
python training/strip_design_sweep.py --sweep-epochs 5 --max-runs 3
```

The sweep trains 72 combinations (3 pooling × 2 padding × 2 background × 3 slice widths × 2 strides). Each result is appended to `outputs/results.csv` as it completes, so the sweep is safe to interrupt and resume.

**Step 3 — Summarise results**

```bash
python evaluation/summarize.py
# or point at a specific results file:
python evaluation/summarize.py --results outputs/results.csv --top 10
```

Prints the top-N configs by val AUC and the marginal effect of each design axis.

---

### 4. Retrieve results

```bash
# From your local machine:
scp -r <user>@<host>:~/fine-grained-homoglyph-detection/outputs/ ./outputs/
```

Or use WinSCP to drag the `outputs/` folder back to your machine.
