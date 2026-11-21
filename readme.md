# Siamese CNN (From Scratch): Multi-Task Face Analysis & Verification

A self-contained PyTorch pipeline for multi-task facial representation learning, demographic classification, verification, and fairness auditing. Built from scratch (~1.37M parameters), the model trains a shared convolutional backbone with **dual $L_2$-normalized projection heads** and **three demographic classification heads**:

1. **Attribute-Pair Verification (`proj_attr`):** Contrastive margin loss learning demographic similarity across `(gender, race, age_bin)`.
2. **Instance Verification (`proj_inst`):** SimCLR-style NT-Xent loss learning augmentation-invariant identity/instance embeddings.
3. **Demographic Classification Heads:** Simultaneous prediction of **Gender** (2 classes), **Race** (5 shared classes or 7 FairFace native classes), and **Age Bin** (9 bins from `0-2` to `70+`).

---

## 1. Environment & Dependencies

The notebook automatically adapts to **Kaggle**, **Google Colab**, or **local Python environments**:

* **Working Directory (`WORK_ROOT`):** Uses `/kaggle/working` when running on Kaggle and automatically falls back to `./working` locally or on Colab.
* **Hardware Acceleration:** Automatically selects CUDA (`DEVICE = "cuda"`) with FP16 Automatic Mixed Precision (`amp=True`) when a GPU is available, or safely falls back to CPU (`amp=False`).

### Installing Dependencies (Local / Colab)
Kaggle pre-installs all required libraries. If running in a fresh local or cloud environment, install:

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn datasets pyarrow
```

*(Optional)* Install OpenCV only if you plan to enable Viola-Jones face cropping (`detect_face=True`) during inference:

```bash
pip install opencv-python
```

---

## 2. Supported Datasets & Label Schemas

| Dataset Key | Source | Race Taxonomy (`race_schema`) | Age Labels |
| :--- | :--- | :--- | :--- |
| `"utkface"` | Local/Kaggle folder of `age_gender_race_*.jpg` files | `"unified_5"` (`White`, `Black`, `Asian`, `Indian`, `Others`) | Continuous years (`1–116`) + 9 age bins |
| `"fairface"` | `HuggingFaceM4/FairFace` (`0.25` or `1.25`) or `.zip` archive | `"native"` (7 classes) or `"unified_5"` (5 mapped classes) | 9 age bins (`0-2` through `70+`) |
| `"combined"` | Concatenated UTKFace + FairFace | `"unified_5"` (required for shared label space) | Continuous (UTKFace) + 9 age bins |

### Setting Up Dataset Sources
* **UTKFace:** Attach the UTKFace dataset on Kaggle (default path: `/kaggle/input/datasets/jangedoo/utkface-new/UTKFace`) or pass your local path via `utk_dir="/path/to/UTKFace"`.
* **FairFace:**
  * `fairface_source="auto"` *(default)*: Searches `/kaggle/input`, `./working`, and `.` for any `*fairface*.zip` archive first; if none is found, automatically downloads `HuggingFaceM4/FairFace` via the Hugging Face `datasets` library.
  * `fairface_source="hf"`: Forces loading directly from `HuggingFaceM4/FairFace`.
  * `fairface_source="/path/to/fairface.zip"`: Extracts and loads from a specific local or uploaded `.zip` file (supports both standard CSV+image zips and Hugging Face Parquet zips).

> **Automatic Schema Syncing:** `make_config()` automatically resolves `race_schema` when switching datasets—selecting `"native"` (7 classes) for FairFace and switching to `"unified_5"` (5 classes) whenever `"utkface"` or `"combined"` is chosen.

---

## 3. How to Use the Notebook

You can run the notebook in **two ways**:

### Option A: Interactive Top-to-Bottom Mode
Ideal for step-by-step inspection, visual sanity checks, and automated `PASS` / `FAIL` invariant testing:
1. Open **Step 0** and set the global toggles or defaults inside `@dataclass class Config`:
   * `dataset`: `"fairface"`, `"utkface"`, or `"combined"`
   * `QUICK_RUN`: `True` (~3,000 images, 2 epochs for fast smoke-testing) or `False` (full dataset training)
   * `Config.run_ablation`: `True` to run the multi-seed `A / B / C` ablation study in **Step 12**
   * `STOP_ON_FAIL`: `True` to halt immediately if any interactive validation check fails
2. Optionally set `UTK_DIR`, `FAIRFACE_SOURCE`, and `FAIRFACE_CONFIG` in **Step 2**.
3. Click **Run All** (or execute cells sequentially from **Step 0** through the **Check summary** cell). Every step prints automated verification results (`PASS` / `FAIL`).

### Option B: Single-Function Programmatic Mode (`run_experiment`)
After running the function definitions (or running the notebook once), execute complete end-to-end experiments via `run_experiment(...)`:

> **Important:** Because `QUICK_RUN = True` is set in Step 0 by default, explicitly pass `quick_run=False` (or set `QUICK_RUN = False` in Step 0) when launching full training runs via `run_experiment(...)`.

```python
# 1. Train on FairFace (native 7-class race taxonomy, official test split)
res_ff = run_experiment(
    dataset="fairface",
    fairface_source="auto",      # "auto", "hf", or "/path/to/fairface.zip"
    fairface_config="0.25",      # "0.25" or "1.25"
    race_schema="native",
    split_mode="official",       # uses FairFace validation split as test
    epochs=15,
    steps_per_epoch=300,
    quick_run=False,
    show_plots=True,
)
display(res_ff["metrics_table"])
display(res_ff["classification_reports"]["race_df"])

# 2. Train on UTKFace
res_utk = run_experiment(
    dataset="utkface",
    utk_dir="/kaggle/input/datasets/jangedoo/utkface-new/UTKFace",
    epochs=20,
    steps_per_epoch=300,
    quick_run=False,
    show_plots=False,
)

# 3. Train on Combined (UTKFace + FairFace)
res_comb = run_experiment(
    dataset="combined",
    epochs=15,
    steps_per_epoch=400,
    quick_run=False,
)

# 4. Cross-Dataset Generalization (train on FairFace, evaluate on UTKFace)
res_cross = run_experiment(
    dataset="fairface",
    eval_dataset="utkface",
    race_schema="unified_5",     # required so train & test share 5 race classes
    epochs=15,
    quick_run=False,
)
```

#### Return Dictionary Keys from `run_experiment(...)`
* `metrics_table`: Unified `pandas.DataFrame` summarizing Accuracy, Balanced Accuracy, Precision, Recall, Macro/Weighted F1, ROC-AUC, PR-AUC, and MCC across all 5 tasks.
* `classification_reports`: Dict containing full reports and per-class DataFrames (`gender_df`, `race_df`, `age_df`).
* `verification`: Attribute-pair and instance verification metrics (ROC-AUC, PR-AUC, EER, confusion matrices, and difficulty breakdown `auc_by_ndiff`).
* `fairness`: Disparity metrics across `by_race_df`, `by_gender_df`, `by_race_gender_df`, plus group-size-equalized `balanced` fairness.
* `bootstrap`: Image-level multinomial bootstrap 95% confidence intervals for accuracies, worst-group accuracies, disparity gaps, and verification AUCs.
* `best_model_path`: Path to the saved `siamese_best.pt` checkpoint.
* `history`: Epoch-by-epoch training and validation metrics `DataFrame`.

---

## 4. Global Variables & Override Precedence

When `make_config(base=None, **overrides)` builds a configuration (both at the bottom of **Step 0** and inside `run_experiment()`), settings are resolved in the following order from lowest to highest priority:

1. **Dataclass Defaults (`Config`):** Base defaults defined in `@dataclass class Config` (e.g., `dataset="fairface"`, `race_schema="native"`, `quick_run=False`).
2. **Active Top-Level Globals (`globals()`):** When `base is None`, `make_config()` inspects the notebook's global namespace and applies any defined variables listed in `global_map`.
3. **Explicit Keyword Overrides (`**overrides`):** Arguments passed directly to `make_config(...)` or `run_experiment(...)` overwrite both `Config` defaults and global variables.
4. **Automatic Schema Alignment (`sync_config_schema`):** If `race_schema` was not explicitly overridden, `make_config()` automatically switches `race_schema` to `"unified_5"` when `dataset` is `"utkface"` or `"combined"`, and to `"native"` when `dataset` is overridden to `"fairface"`.

### Status of Global Variables in Steps 0 & 2

| Variable | Location | Status | Behavior & Notes |
| :--- | :--- | :--- | :--- |
| `QUICK_RUN` | Step 0 | **Active** | Mapped to `Config.quick_run` via `make_config()`. Because it defaults to `True` in Step 0, it overrides `Config.quick_run=False` unless you set `QUICK_RUN = False` or pass `quick_run=False` to `run_experiment()`. |
| `STOP_ON_FAIL` | Step 0 | **Interactive only** | Controls `DEFAULT_TRACKER` and `check()` during sequential interactive cell execution. `run_experiment()` uses `Config.stop_on_fail` instead. |
| `RUN_ABLATION` | Step 0 | **Disconnected** | Defined in Step 0 (`RUN_ABLATION = False`), but not included in `make_config()`'s `global_map`. Step 12 checks `cfg.run_ablation`, so pass `run_ablation=True` to `run_experiment()` or edit `Config.run_ablation`. |
| `USE_FAIRFACE` | Step 0 | **Deprecated** | Legacy switch kept only for backward compatibility; ignored by `make_config()` and all downstream steps. Use `dataset="fairface"` or `"combined"`. |
| `DATASET`, `RACE_SCHEMA`, `SPLIT_MODE`, `EVAL_DATASET` | Optional (Step 0) | **Supported** | Not declared in Step 0 by default, but if you define any of these uppercase variables in the global scope, `make_config()` will automatically pick them up. |
| `UTK_DIR`, `FAIRFACE_SOURCE`, `FAIRFACE_CONFIG` | Step 2 | **Interactive only** | Copies values from `cfg` and writes them back to `cfg` during Step 2 for interactive runs; bypassed when calling `run_experiment()`. |

---

## 5. Key Configuration Reference (`Config`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `dataset` | `"fairface"` | Primary dataset: `"utkface"`, `"fairface"`, or `"combined"`. |
| `eval_dataset` | `None` | Optional cross-dataset evaluation target (`"utkface"` or `"fairface"`). |
| `race_schema` | `"native"` | `"native"` (FairFace 7 classes) or `"unified_5"` (5 shared classes). |
| `split_mode` | `"stratified"` | `"stratified"` (70/15/15 split) or `"official"` (uses dataset's official val/test split). |
| `drop_near_dups` | `False` | If `True`, drops val/test images within `near_dup_max_dist` Hamming bits of train. |
| `img_size` | `64` | Input resolution ($64 \times 64$ RGB). |
| `epochs` | `15` | Maximum training epochs (with `OneCycleLR` scheduler). |
| `steps_per_epoch` | `300` | Optimization steps per epoch (`None` = full pass over training set). |
| `early_stop_patience` | `4` | Stop training if `val_loss` does not improve for $N$ epochs (`0` = disabled). |
| `w_pair` / `w_inst` / `attr_weight` | `1.0 / 1.0 / 1.0` | Loss weights for attribute contrastive loss, NT-Xent instance loss, and classification heads. |
| `quick_run` | `False` | Subsamples `quick_total` (3,000) images for fast end-to-end testing (overridden by Step 0 `QUICK_RUN` unless explicitly passed). |
| `resume` | `False` | Resumes training from `last.pt` in the run's output folder if interrupted. |

---

## 6. Output Directory Structure

Each training configuration saves into its own isolated subfolder tagged by `<dataset>_<race_schema>_s<seed>` so runs never overwrite one another:

```text
working/
├── cache/
│   ├── cache_<md5_hash>.npy              # Disk-cached decoded uint8 image tensors
│   └── fairface_hf_0_25/                 # Cached decoded FairFace images
└── siamese_cnn/
    └── C_full/
        └── fairface_native_s42/
            ├── siamese_best.pt           # Self-contained inference checkpoint (weights + metadata)
            ├── best.pt                   # Best validation checkpoint
            ├── last.pt                   # Latest epoch checkpoint (for resume=True)
            ├── results.json              # Complete serialized metrics, CIs, and fairness report
            ├── test_embeddings_attr.npy  # Test set attribute embeddings (N x 128)
            └── test_embeddings_inst.npy  # Test set instance embeddings (N x 128)
```

When `run_ablation=True` (Step 12), multi-seed comparisons for `A_heads_only`, `B_pairs_only`, and `C_full` are logged incrementally to `working/siamese_cnn/ablation/ablation_results.csv`.

---

## 7. Inference & Pair Comparison (`run_siamese_pipeline`)

Use `run_siamese_pipeline` (Step 13) to run single-image demographic predictions or compare two face images using any saved `siamese_best.pt` checkpoint. The checkpoint automatically restores the exact architecture (`n_race`, `race_names`, `img_size`) and validation-calibrated distance thresholds:

```python
ckpt_path = res_ff["best_model_path"]

# 1. Single-image demographic prediction (accepts file path, PIL Image, or RGB numpy array)
pred = run_siamese_pipeline(
    image="path/to/face.jpg",
    checkpoint=ckpt_path,
    detect_face=False,          # Set True to auto-crop largest face via OpenCV Haar cascade
    include_embeddings=False,   # Set True to return 128-d L2-normalized embedding vectors
)
print(pred["prediction"])
# -> {'gender': 'female', 'gender_confidence': 0.94, 'race': 'East Asian',
#     'race_probabilities': {...}, 'age_bin': '20-29', 'age_expected_years': 26.4}

# 2. Pairwise demographic & instance comparison
comp = run_siamese_pipeline(
    image="path/to/person_a.jpg",
    reference="path/to/person_b.jpg",
    checkpoint=ckpt_path,
)
print(comp["comparison"])
# -> {'attr_distance': 0.31, 'attr_threshold': 0.52, 'same_group': True,
#     'inst_distance': 1.18, 'inst_threshold': 0.65, 'same_image': False}
```