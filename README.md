# Modal Guidance: Towards Balanced Fusion via Staged Distillation

Yuqi Wang, Haijian Liu, Yifan Fu, Haoyu Yin, Min Xia, Liguo Weng, and Haifeng Lin

Nanjing University of Information Science and Technology · Nanjing Forestry University

## 👀 Introduction

This repository provides the PyTorch implementation of **LFnet**, a multimodal remote sensing semantic segmentation network, and **CL-CMFF**, its staged training framework. CL-CMFF uses weight inheritance and supervision from an optical teacher to help the auxiliary modality X learn segmentation-relevant features before fusion. The optical and X teachers then jointly supervise fusion training.

LFnet uses two independent shallow encoders for optical and X features. A shared deep encoder and decoder produce the segmentation predictions after fusion.

- **SAMT (Stagewise Adaptive Modal Truncation)** controls the active paths and parameter updates at each training stage.
- **SALI (Stream Attention and Learned Integration)** fuses the two streams through deformable sampling, gated aggregation, and bidirectional cross-linear attention.
- **SGFS (Saliency-Guided Feature Selection)** groups and weights deep features by their responses before combining them with shallow features in the decoder.

## 💡 Environment

We recommend Linux or WSL2 with an NVIDIA GPU. The reference environment uses Python 3.10 and PyTorch 2.9.1 with CUDA 12.8. Choose compatible versions for your GPU, driver, and CUDA setup. To use the reference environment, run the following commands from the repository root.

```bash
conda env create -f environment.yml
conda activate work1
```

The supplied configurations enable BF16 by default. Set `system.use_bf16` to `false` if your GPU lacks BF16 support.

## ⏳ Setup

### Datasets

| Dataset | Optical Input | Auxiliary Input | Configuration |
| --- | --- | --- | --- |
| [ISPRS Vaihingen](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | IRRG | DSM | [Vaihingen_T.yaml](configs/Vaihingen/Vaihingen_T.yaml) |
| [ISPRS Potsdam](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | RGB | NIR + nDSM | [Potsdam_T.yaml](configs/Potsdam/Potsdam_T.yaml) |
| [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset) | RGB | NIR + SAR | [WHU_T.yaml](configs/WHU/WHU_T.yaml) |

The training, validation, and test splits use fixed tile IDs. Model inputs are 512 × 512 patches. When there are multiple auxiliary inputs, concatenate them along the channel dimension and feed them into the same X branch.

To prepare patches from raw imagery, set `RAW_DATA_ROOT` and `OUTPUT_ROOT` in the corresponding script under [data_split/](data_split/). Each script defines the source filenames and tile IDs for each split. Use a separate output directory for generated patches.

```bash
python data_split/Vaihingen.py
python data_split/Potsdam.py
python data_split/WHU.py
```

If you already have patches, set `data.root_dir` in the YAML to the dataset's split directory. Within each split, the optical image, auxiliary inputs, and mask for a sample use the same `.tif` filename. Masks store integer class IDs.

```text
dataset/
├── Vaihingen_split/
│   └── {train,val,test}/{images,dsm,masks}/
├── Potsdam_split/
│   └── {train,val,test}/{images,ir,dsm,masks}/
└── WHU_split/
    └── {train,val,test}/{images,ir,sar,masks}/
```

### Pretrained Backbone

The paper uses the name MILA-T, and the configuration uses `MLLA-T` as the identifier. Download the MLLA-T checkpoint and place it at `pretrain/MLLA-T.pth`, or set `model.pretrained_path` to your existing copy. Evaluation loads the trained segmentation checkpoint directly. [Official MLLA model zoo](https://github.com/LeapLabTHU/MLLA/blob/master/README.md)

## 📦 Usage

### Training

Run the command for your dataset from the repository root.

**Vaihingen**

```bash
torchrun --nproc_per_node=1 --master_port=4567 train_ddp.py --config configs/Vaihingen/Vaihingen_T.yaml
```

**Potsdam**

```bash
torchrun --nproc_per_node=2 --master_port=4568 train_ddp.py --config configs/Potsdam/Potsdam_T.yaml
```

**WHU**

```bash
torchrun --nproc_per_node=2 --master_port=4569 train_ddp.py --config configs/WHU/WHU_T.yaml
```

The default GPU IDs are `[0]` for Vaihingen, `[3, 4]` for Potsdam, and `[5, 6]` for WHU. Adjust `system.device_ids` in the YAML for your machine and set `--nproc_per_node` to the number of listed GPUs.

Each command runs the three stages for 45, 65, and 50 epochs, respectively. The default base seed is 42 and the batch size is 8 per GPU.

| Stage | Mode | Training |
| --- | --- | --- |
| 1 | `f1` | Train the optical branch with an exponential moving average (EMA) teacher. |
| 2 | `f2` | Inherit optical weights from Stage 1 and train the auxiliary branch under a frozen optical teacher. |
| 3 | `all` | Freeze both shallow encoders by default and train the fusion modules, deep encoder, and decoder with prediction- and feature-level supervision from the optical and X teachers. |

The configuration copy, training log, and stage checkpoints are saved under `weight/<system.weight_dir_name>/<run_id>/`. Each stage selects its best checkpoint by validation mIoU after reaching the configured saving threshold. `best_manifest.json` records the epoch, validation mIoU, and checkpoint hashes.

To resume the latest unfinished run, append `--continue` to the original command and use the same configuration and process count. Training resumes from the last completed epoch. If no unfinished run has a recovery checkpoint, the script starts a new run.

### Evaluation

Trained LFnet checkpoints are publicly available. [Download the trained weights from Google Drive](https://drive.google.com/drive/folders/1599qnAEGMo800db0ynD7U6DcFA4O7U03?usp=sharing)

The following commands evaluate the Stage 3 model on the test set. Replace the example path after `--weight` with the actual checkpoint for the corresponding dataset. Evaluation uses GPU 0 by default.

**Vaihingen**

```bash
python val.py --config configs/Vaihingen/Vaihingen_T.yaml --weight /path/to/Vaihingen/stage3/best_model.pth
```

**Potsdam**

```bash
python val.py --config configs/Potsdam/Potsdam_T.yaml --weight /path/to/Potsdam/stage3/best_model.pth
```

**WHU**

```bash
python val.py --config configs/WHU/WHU_T.yaml --weight /path/to/WHU/stage3/best_model.pth
```

If you changed the configuration for training, pass the YAML saved with that run to `--config`. To evaluate Stage 1 or Stage 2, use the corresponding checkpoint and add `--mode f1` or `--mode f2`, respectively.

Use `vis.py` to save metrics, a confusion matrix, and prediction images.

```bash
python vis.py --config configs/Vaihingen/Vaihingen_T.yaml --weight /path/to/Vaihingen/stage3/best_model.pth
```

Add `--split val` to evaluate the validation set with `vis.py`.

### Pre-Fusion L2 Norm Statistics

Run [`l2_stats.py`](l2_stats.py) from the repository root with the dataset configuration and Stage 3 fusion checkpoint.

```bash
python l2_stats.py --config configs/Vaihingen/Vaihingen_T.yaml --weight /path/to/Vaihingen/stage3/best_model.pth --split test
```

- `--split` accepts `train`, `val`, or `test` and defaults to `test`. Use `--split all` to process all three splits.
- The script reads the data path from `data.root_dir`. Use `--data-root /path/to/Vaihingen_split` to override it and `--normalization /path/to/normalization.json` to reuse saved training-set normalization statistics.
- The defaults are GPU 0 and a batch size of 2. The script computes L2 norms and the ratio per batch, then averages over batches with equal weight, including the final incomplete batch.

The script writes three CSV rows per split, one for each of SALI-1, SALI-2, and SALI-3. The main fields are the optical feature norm `O_L2`, auxiliary feature norm `X_L2`, and auxiliary feature-magnitude share `RatioX_percent`. Append `> l2_stats.csv` to save the output.

## 📈 Results

The table reports the revised manuscript's Stage 3 test results. Each dataset has three complete runs with a base seed of 42. Within each run, checkpoints are selected by validation mIoU. The manuscript's main tables report the run with the highest Stage 3 test mIoU for each dataset. Because the test set was used to select among runs, the headline scores describe the selected runs.

| Dataset | Run ID | mIoU (%) | mF1 (%) | OA (%) | Three-Run mIoU (Mean ± Sample SD, %) |
| --- | --- | ---: | ---: | ---: | ---: |
| Vaihingen | `20261004_165310_064080` | 87.32 | 93.12 | 92.69 | 87.18 ± 0.23 |
| Potsdam | `20261004_164929_287764` | 88.93 | 94.02 | 92.63 | 88.87 ± 0.06 |
| WHU-OPT-SAR | `20261005_234136_210861` | 59.26 | 72.97 | 83.40 | 59.19 ± 0.06 |

Per-class metrics are computed from the full confusion matrix. mIoU and mF1 average the first five classes for Vaihingen and Potsdam and the first seven for WHU-OPT-SAR. OA includes all valid labels. Label 255 is ignored. Misclassifications between the reported classes and clutter or background remain in the confusion matrix.

### Representation and Robustness Analysis

- **Frozen shallow-feature probes.** Compared with single-stage fusion training, staged training improves the segmentation decodability of X features on all three datasets. Linear probe mIoU increases by 8.39, 6.01, and 0.45 percentage points on Potsdam, Vaihingen, and WHU-OPT-SAR, respectively.
- **Pre-fusion feature magnitude.** At the deepest SALI input, staged training brings the optical and X feature L2 norms closer on all three datasets.
- **Modality masking and spatial perturbations.** The revised manuscript reports inference-time input masking and tests translations and rotations at each of the three fusion scales. Compared with a model that uses concatenation and projection in place of SALI, the full model shows a smaller perturbation-induced mIoU drop on all three datasets.

### Model Efficiency

With the Potsdam configuration, LFnet has 35.47M parameters and uses 71.04G FLOPs per 512 × 512 input. On one RTX 3090 with batch size 8 and FP32, forward-pass throughput is 41.7 FPS. The measurement excludes data loading and post-processing.

## 🙏 Acknowledgements

We thank the authors of MILA (MLLA) for releasing their backbone and pretrained weights, and the ISPRS and WHU-OPT-SAR teams for providing their datasets. [MILA (MLLA)](https://github.com/LeapLabTHU/MLLA) · [ISPRS benchmark](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) · [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset)

## 📌 Citation

```bibtex
@unpublished{wang2026modalguidance,
  title  = {Modal Guidance: Towards Balanced Fusion via Staged Distillation},
  author = {Wang, Yuqi and Liu, Haijian and Fu, Yifan and Yin, Haoyu and Xia, Min and Weng, Liguo and Lin, Haifeng},
  year   = {2026},
  note   = {Manuscript}
}
```
