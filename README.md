# Modal Guidance: Towards Balanced Fusion via Staged Distillation

Yuqi Wang, Haijian Liu, Yifan Fu, Haoyu Yin, Min Xia, Liguo Weng, and Haifeng Lin

南京信息工程大学 · 南京林业大学

## 👀 简介

本仓库提供多模态遥感语义分割网络 **LFnet** 及其分阶段训练框架 **CL-CMFF** 的 PyTorch 实现。光学分支可以利用预训练先验。CL-CMFF 通过权重继承和教师监督，让辅助模态 X 在融合前学习与分割相关的特征，再由光学和 X 两个教师共同指导融合训练。

LFnet 用两个独立的浅层编码器分别提取光学和 X 特征，融合后由共享的深层编码器和解码器生成分割结果。

- **SAMT（分阶段自适应模态截断）**控制各阶段的激活路径和参数更新。
- **SALI（流注意力与可学习集成）**通过可变形采样、门控聚合和双向交叉线性注意力融合两路特征。
- **SGFS（显著性引导特征选择）**在解码阶段按响应分组并加权深层特征，再与浅层特征结合。

## 💡 环境

建议使用配有 NVIDIA GPU 的 Linux 或 WSL2 环境。复现实验可参考 Python 3.10 和 PyTorch 2.9.1（CUDA 12.8 构建版），也可根据本机 GPU、驱动和 CUDA 环境选择兼容版本。若希望使用仓库提供的参考环境，可在仓库根目录执行以下命令。

```bash
conda env create -f environment.yml
conda activate work1
```

仓库配置默认启用 BF16。若 GPU 不支持 BF16，将 `system.use_bf16` 设为 `false`。

## ⏳ 准备工作

### 数据集

| 数据集 | 主分支输入 | 辅助分支输入 | 配置文件 |
| --- | --- | --- | --- |
| [ISPRS Vaihingen](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | IRRG | DSM | [Vaihingen_T.yaml](configs/Vaihingen/Vaihingen_T.yaml) |
| [ISPRS Potsdam](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | RGB | NIR + nDSM | [Potsdam_T.yaml](configs/Potsdam/Potsdam_T.yaml) |
| [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset) | RGB | NIR + SAR | [WHU_T.yaml](configs/WHU/WHU_T.yaml) |

三个数据集均按固定影像编号划分训练、验证和测试集，模型输入为 512 × 512 切片。多个辅助通道沿通道维拼接后送入同一个 X 分支。

从原始影像制备数据时，在 [data_split/](data_split/) 对应脚本中设置原始数据根目录 `RAW_DATA_ROOT` 和切片输出目录 `OUTPUT_ROOT`。各划分的源文件命名规则和影像编号已在脚本中定义。生成的切片应写入独立的输出目录。

```bash
python data_split/Vaihingen.py
python data_split/Potsdam.py
python data_split/WHU.py
```

如果已有切片数据，将 YAML 中的 `data.root_dir` 指向对应的切分数据根目录。默认目录结构如下。各划分内，同一样本的主模态影像、辅助模态影像和标签使用相同的 `.tif` 文件名。标签像素存储整数类别编号。

```text
dataset/
├── Vaihingen_split/
│   └── {train,val,test}/{images,dsm,masks}/
├── Potsdam_split/
│   └── {train,val,test}/{images,ir,dsm,masks}/
└── WHU_split/
    └── {train,val,test}/{images,ir,sar,masks}/
```

### 骨干网络预训练权重

论文将骨干网络记为 MILA-T，仓库配置沿用 `MLLA-T` 作为标识。从 [MLLA 官方模型库](https://github.com/LeapLabTHU/MLLA/blob/master/README.md) 下载 MLLA-T 权重。仓库配置默认读取 `pretrain/MLLA-T.pth`。如果已有该权重，将 `model.pretrained_path` 指向现有文件即可。评估时直接加载训练好的分割模型权重。

## 📦 使用方法

### 训练

在仓库根目录执行对应数据集的命令。

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

默认 GPU 编号分别为 Vaihingen `[0]`、Potsdam `[3, 4]`、WHU `[5, 6]`。按机器调整 YAML 中的 `system.device_ids`，并让 `--nproc_per_node` 等于列表中的 GPU 数量。

每条命令依次执行三个阶段，训练轮数分别为 45、65 和 50。默认基础随机种子为 42，每张 GPU 的批大小为 8。

| 阶段 | 模式 | 训练内容 |
| --- | --- | --- |
| 1 | `f1` | 使用指数移动平均（EMA）教师训练主光学分支。 |
| 2 | `f2` | 从第一阶段继承光学权重，用冻结的光学教师指导辅助模态分支学习。 |
| 3 | `all` | 默认冻结两个浅层编码器，在光学和 X 教师的预测级与特征级监督下训练融合模块、深层编码器和解码器。 |

配置副本、训练日志和各阶段的权重保存在 `weight/<system.weight_dir_name>/<run_id>/`。每个阶段达到配置指定的保存起始轮次后，按验证集 mIoU 选取最佳权重。`best_manifest.json` 记录权重对应的训练轮次、验证集 mIoU 和文件哈希。

如需恢复最近一次未完成的训练，在原命令末尾添加 `--continue`，并沿用原配置和进程数。程序从最近一个已完成的训练轮次接续。如果没有带恢复点的未完成运行，程序会启动一次新训练。

### 评估

训练好的 LFnet 权重可公开下载。[在 Google Drive 下载训练权重](https://drive.google.com/drive/folders/1599qnAEGMo800db0ynD7U6DcFA4O7U03?usp=sharing)

以下命令在测试集上评估第三阶段模型。将 `--weight` 后的示例路径换成对应数据集的第三阶段权重文件路径。评估默认使用 GPU 0。

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

如果训练时修改过配置，`--config` 指向该次训练保存的 YAML。评估第一、第二阶段时，分别使用对应阶段的权重，并添加 `--mode f1` 或 `--mode f2`。

需要保存指标、混淆矩阵和预测图时，可用 `vis.py`。例如：

```bash
python vis.py --config configs/Vaihingen/Vaihingen_T.yaml --weight /path/to/Vaihingen/stage3/best_model.pth
```

给 `vis.py` 加上 `--split val` 可评估验证集。

### 融合前 L2 范数统计

在仓库根目录运行 [`l2_stats.py`](l2_stats.py)。传入对应数据集的配置和第三阶段融合权重，下面以 Vaihingen 为例。

```bash
python l2_stats.py --config configs/Vaihingen/Vaihingen_T.yaml --weight /path/to/Vaihingen/stage3/best_model.pth --split test
```

- `--split` 可选 `train`、`val`、`test`，默认是 `test`。用 `--split all` 一次统计三个划分。
- 数据路径默认读取配置中的 `data.root_dir`。如果切片数据在其他位置，添加 `--data-root /path/to/Vaihingen_split`。已有训练集归一化统计时，可通过 `--normalization /path/to/normalization.json` 复用。
- 默认使用 GPU 0，批大小为 2。脚本对每批计算 L2 范数和比例，再对批次等权平均，包含最后一个不满批次。

每个划分输出 SALI-1、SALI-2、SALI-3 三行 CSV。主要指标是光学特征范数 `O_L2`、辅助模态特征范数 `X_L2` 和辅助模态幅度占比 `RatioX_percent`。在命令末尾加 `> l2_stats.csv` 即可保存结果。

## 📈 实验结果

下表对应修订稿的 LFnet 第三阶段测试结果。每个数据集完成三次基础随机种子为 42 的训练。每次运行内部按验证集 mIoU 选取各阶段的权重，修订稿主表按第三阶段测试集 mIoU 选取运行。测试集参与了跨运行筛选，主表数值应按最佳运行结果解读。

| 数据集 | 运行编号 | mIoU (%) | mF1 (%) | OA (%) | 三次运行 mIoU（均值 ± 样本标准差，%） |
| --- | --- | ---: | ---: | ---: | ---: |
| Vaihingen | `20261004_165310_064080` | 87.32 | 93.12 | 92.69 | 87.18 ± 0.23 |
| Potsdam | `20261004_164929_287764` | 88.93 | 94.02 | 92.63 | 88.87 ± 0.06 |
| WHU-OPT-SAR | `20261005_234136_210861` | 59.26 | 72.97 | 83.40 | 59.19 ± 0.06 |

mIoU 和 mF1 在完整混淆矩阵上计算各类指标，再对 Vaihingen 和 Potsdam 的前 5 类、WHU-OPT-SAR 的前 7 类取平均。OA 统计所有有效标签，忽略值 255 不参与计算。杂物或背景类别与其余类别之间的误判仍计入混淆矩阵。

### 表征与鲁棒性分析

- **冻结浅层特征探针**。相较单阶段融合训练，三个数据集的 X 分支特征都能被线性分割探针更准确地解码。Potsdam、Vaihingen 和 WHU-OPT-SAR 的探针 mIoU 分别提高 8.39、6.01 和 0.45 个百分点。
- **融合前特征幅度**。在三个数据集最深的 SALI 输入处，分阶段训练使光学与 X 特征的 L2 范数更接近。
- **模态遮蔽与空间扰动**。修订稿报告了推理时遮蔽输入通道的影响，并在三个融合尺度分别施加平移和旋转。相较以拼接和投影替代 SALI 的模型，完整模型在三个数据集上的扰动后 mIoU 降幅更小。

### 模型效率

在 Potsdam 配置下，LFnet 有 35.47M 参数，单张 512 × 512 输入的计算量为 71.04G FLOPs。在单张 RTX 3090 上使用批大小 8、FP32 推理，纯前向吞吐率为 41.7 FPS。该测量排除数据加载和后处理。

## 🙏 致谢

感谢 [MILA（MLLA）作者](https://github.com/LeapLabTHU/MLLA)公开骨干网络和预训练权重，也感谢 [ISPRS 基准](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)及 [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset)团队提供数据集。

## 📌 引用

```bibtex
@unpublished{wang2026modalguidance,
  title  = {Modal Guidance: Towards Balanced Fusion via Staged Distillation},
  author = {Wang, Yuqi and Liu, Haijian and Fu, Yifan and Yin, Haoyu and Xia, Min and Weng, Liguo and Lin, Haifeng},
  year   = {2026},
  note   = {Manuscript}
}
```
