# BOFT DreamBooth Mini-Project

本项目基于 `boft.ipynb`，使用 **BOFT (Butterfly Orthogonal Fine-Tuning)** 对预训练 Stable Diffusion 模型进行 DreamBooth 主体驱动生成微调（subject-driven generation）。

## 1. 项目目标（对应 mini-project 要求）

- 选择一个下游任务：DreamBooth 主体定制生成
- 使用预训练基础模型：`sd2-community/stable-diffusion-2-1`
- 使用参数高效微调方法：BOFT（PEFT）
- 给出微调前后可视化对比与结果分析

## 2. 方法简介

BOFT 在 UNet 注意力层中注入带蝶形分解结构的正交可训练矩阵，仅训练适配器参数，冻结原始大模型参数。核心特点：

- 参数高效：仅训练很小比例参数（运行时会打印 trainable ratio）
- 正交约束：尽量保持预训练表示空间结构
- 全秩更新：相比低秩加性方法，表达能力更强

参考资料：
- [BOFT Paper (ICLR 2024)](https://arxiv.org/abs/2311.06243)
- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [PEFT 文档](https://huggingface.co/docs/peft)

## 3. 当前代码与目录（已对齐实际仓库）

```text
BOFT/
├── boft.ipynb
├── environment.yml
├── README.md
├── baseline_images.png
├── finetuned_images.png
├── comparison_before_after.png
├── multi_prompt_results.png
├── training_loss.png
├── data/
│   ├── dreambooth/dataset/dog/
│   ├── class_data/dog/
│   └── output/boft/unet/
│       ├── 200/
│       ├── 400/
│       ├── 600/
│       └── 800/
└── report/
    └── report.md
```

## 4. 实验配置（来自 `boft.ipynb`）

### 模型与任务

- Base model: `sd2-community/stable-diffusion-2-1`
- 唯一标识词：`sks`
- 实例提示词：`a photo of sks dog`
- 类别提示词：`a photo of dog`

### 训练参数

- `RESOLUTION=512`
- `MAX_TRAIN_STEPS=800`
- `LEARNING_RATE=3e-5`
- `TRAIN_BATCH_SIZE=1`
- `NUM_CLASS_IMAGES=100`
- `CHECKPOINT_STEPS=200`
- `PRIOR_LOSS_WEIGHT=1.0`

### BOFT 参数

- `boft_block_num=8`
- `boft_block_size=0`
- `boft_n_butterfly_factor=1`
- `boft_dropout=0.1`
- `target_modules=[to_q, to_v, to_k, to_out.0]`
- `bias=none`（见 `data/output/boft/unet/800/adapter_config.json`）

## 5. 运行方式

### 5.1 创建环境

```bash
conda env create -f environment.yml
conda activate aist
```

### 5.2 启动 Notebook

```bash
jupyter notebook boft.ipynb
```

按顺序执行单元格即可完成：
1. 数据准备与可视化
2. 微调前基线生成
3. BOFT 微调训练（含 checkpoint）
4. 损失曲线绘制
5. 微调后生成与定性对比
6. 多提示词泛化生成

## 6. 结果文件

- 基线图：`baseline_images.png`
- 微调后图：`finetuned_images.png`
- 前后对比图：`comparison_before_after.png`
- 多提示词结果：`multi_prompt_results.png`
- 训练损失图：`training_loss.png`
- 适配器 checkpoint：`data/output/boft/unet/{200,400,600,800}`

## 7. 说明

- 当前仓库以 Notebook 实现为主，不包含独立 `train_dreambooth.py`。
- `report/report.md` 已根据本仓库实际配置与产物同步更新。
