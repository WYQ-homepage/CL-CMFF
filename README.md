# Reproducing CL-CMFF Evaluation Metrics

This repository accompanies the **Algorithm Self-Audit** in the response letter for *Modal Guidance: Towards Balanced Fusion via Staged Distillation*.

## 👀 Introduction

This repository reproduces the effect of a confusion-matrix cropping error in the original LFnet evaluator. For each dataset, the script loads one Stage 3 checkpoint, runs inference once on the full test set, and computes both the original and corrected metrics from the same confusion matrix.

The original `val.py` cropped the confusion matrix before computing true positives (TP), false positives (FP), and false negatives (FN).

```python
cm = cm[:valid_classes, :valid_classes]
```

Cropping discards misclassifications between the reported classes and clutter or background. This inflates IoU and F1 for the reported classes and changes which pixels count toward overall accuracy (OA). The corrected evaluator computes per-class metrics from the full confusion matrix and then averages over the designated classes. The macro averages cover five classes for Potsdam and Vaihingen and seven classes for WHU-OPT-SAR, including Other. Corrected OA includes all valid labeled pixels. The evaluator ignores label 255 and labels outside the model's class range.

## 💡 Environment

We recommend Linux or WSL2 with an NVIDIA GPU. The reference environment uses Python 3.10, PyTorch 2.9.1, and torchvision 0.24.1. You can choose compatible versions for your GPU, driver, and CUDA setup. To install the reference dependencies and run the metric tests, use the following commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m tools.test_metrics
```

## ⏳ Setup

### Model Checkpoints

Download the checkpoints used in the response letter for Potsdam, Vaihingen, and WHU-OPT-SAR. [Download from Google Drive](https://drive.google.com/drive/folders/1I6DcawuOpxvIs69EVXTzXh5iw5hnHYiH?usp=sharing)

The evaluator reads the three Stage 3 checkpoints from `weights/` by default. If the downloaded run directories are under `weight/`, run the following commands from the repository root.

```bash
mkdir -p weights
cp -n weight/Potsdam/20260202_174321/phase3_best_model.pth weights/Potsdam.pth
cp -n weight/Vaihingen/20260114_101319/phase3_best_model.pth weights/Vaihingen.pth
cp -n weight/WHU/20260114_211104/phase3_best_model.pth weights/WHU.pth
```

The copy commands preserve existing files with the same names. To use a different directory, prepare the three checkpoints there with the same destination filenames and pass the directory through `--weights`.

### Datasets

| Dataset | Test Patch Directory | Required Subdirectories |
| --- | --- | --- |
| Potsdam | `Potsdam_split/test/` | `images/`, `ir/`, `dsm/`, `masks/` |
| Vaihingen | `Vaihingen_split/test/` | `images/`, `dsm/`, `masks/` |
| WHU-OPT-SAR | `WHU_split/test/` | `images/`, `ir/`, `sar/`, `masks/` |

Each sample uses the same `.tif` filename across its subdirectories. If the test patches are already prepared, point `--data-root` to the parent directory containing the `*_split` directories.

<details>
<summary>Generate test patches from raw imagery</summary>

Arrange the raw data under `dataset/` in the repository root.

| Dataset | Subdirectory | Filename Pattern |
| --- | --- | --- |
| Potsdam | `Potsdam/4_Ortho_RGBIR/` | `top_potsdam_<tile>_RGBIR.tif` |
| Potsdam | `Potsdam/1_DSM_normalisation/` | `dsm_potsdam_<tile>_normalized_lastools.jpg` or `.tif` (zero-padded tile IDs are supported) |
| Potsdam | `Potsdam/5_Labels_all_noBoundary/` | `top_potsdam_<tile>_label_noBoundary.tif` |
| Vaihingen | `Vaihingen/top/` | `top_mosaic_09cm_area<tile>.tif` |
| Vaihingen | `Vaihingen/dsm/` | `dsm_09cm_matching_area<tile>.tif` |
| Vaihingen | `Vaihingen/gts_eroded_for_participants/` | `top_mosaic_09cm_area<tile>_noBoundary.tif` |
| WHU | `whu-opt-sar/optical/` | `<tile>.tif` (four optical channels) |
| WHU | `whu-opt-sar/sar/` | `<tile>.tif` |
| WHU | `whu-opt-sar/lbl/` | `<tile>.tif` (labels 0, 10, …, 70) |

```bash
python -m data_split.split_dataset
```

The script uses fixed test tile IDs and extracts 512 × 512 patches with a stride of 512. The last full window covers each tile edge. Potsdam and Vaihingen use test labels with eroded boundaries. For WHU, the script maps the original labels 10–70 to classes 0–6 and background label 0 to class 7.

</details>

## 📦 Usage

### Evaluate the Full Test Sets

This example uses test patches already prepared in a local WSL environment. Set `--data-root` to the actual data location and use a separate output directory for each evaluation. You can omit `--data-root` if the split directories are under the repository's `dataset/` directory.

```bash
python val.py --data-root /root/work/dataset --weights weights --output results/response_letter --device cuda:0
```

To evaluate a single dataset, add `--datasets Potsdam`, `--datasets Vaihingen`, or `--datasets WHU`. Inference defaults to a batch size of 2 and uses FP32 with TF32 disabled. The evaluator reads fixed training-set normalization statistics from `configs/` and loads checkpoint parameters with `strict=True`.

### Recompute Metrics from a Saved Confusion Matrix

```bash
python -m utils.metrics --dataset Potsdam --matrix results/response_letter/test/Potsdam/confusion_full.npy --output results/recomputed/Potsdam
```

## 📈 Results

The table reports results from the full test sets using the checkpoints from the response letter. The change is the corrected score minus the original score, calculated before rounding.

| Dataset | Test Patches | Original mIoU (%) | Corrected mIoU (%) | Change (pp) |
| --- | ---: | ---: | ---: | ---: |
| Potsdam | 1,152 | 90.33 | 88.95 | −1.37 |
| Vaihingen | 121 | 87.03 | 86.99 | −0.04 |
| WHU-OPT-SAR | 1,760 | 60.44 | 59.56 | −0.88 |

The example writes overall metrics to `results/response_letter/test/summary.csv`. Each dataset subdirectory contains per-class metrics, the full and cropped confusion matrices, and a `comparison.json` file that records the SHA-256 of the checkpoint used and the test sample count.
