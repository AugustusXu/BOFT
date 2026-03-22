# Mini-Project Report: BOFT 用于 DreamBooth 主体驱动生成

**课程主题：** Parameter-Efficient Finetuning for Pretrained Models  
**下游任务：** Subject-Driven Image Generation (DreamBooth)  
**实现文件：** `boft.ipynb`  
**仓库：** `BOFT` 当前目录版本  

## 1. 摘要

本项目基于 Hugging Face `peft` 的 BOFT（Butterfly Orthogonal Fine-Tuning）方法，在 DreamBooth 任务上对 Stable Diffusion 进行参数高效微调。实验采用唯一标识 token `sks` 与类别 `dog` 进行主体绑定，目标是在保持模型通用生成能力的同时学习特定个体外观。项目完整实现于 `boft.ipynb`，并输出了训练损失曲线、微调前后对比图与多提示词生成结果。结果表明，BOFT 适配器能够在较少可训练参数下实现稳定训练和有效个体建模。

## 2. 任务与方法

### 2.1 任务定义

给定少量实例图像（dog 主体），学习文本触发词 `sks dog` 与该主体外观之间的映射，实现：
- 指定主体身份保真（identity preservation）
- 文本指令可控（prompt following）
- 对预训练分布的灾难性遗忘尽量降低（结合 prior preservation）

### 2.2 BOFT 方法简述

BOFT 在 UNet 注意力层引入正交约束的可训练变换。设预训练权重为 $\mathbf{W}_0$，BOFT 通过可学习正交变换 $\mathbf{R}$ 得到

$$
\mathbf{W}=\mathbf{R}\mathbf{W}_0,\quad \mathbf{R}^T\mathbf{R}=\mathbf{I}
$$

相比全量微调，BOFT 仅训练适配器参数；相比常见低秩加性方法，BOFT 通过乘性正交变换保留预训练空间几何结构。

## 3. 实验设置（严格对应 `boft.ipynb`）

### 3.1 模型与数据

- **Base Model:** `sd2-community/stable-diffusion-2-1`
- **Instance Prompt:** `a photo of sks dog`
- **Class Prompt:** `a photo of dog`
- **实例图路径:** `./data/dreambooth/dataset/dog`
- **类别图路径:** `./data/class_data/dog`
- **类别图生成数:** `NUM_CLASS_IMAGES=100`

### 3.2 BOFT 注入位置与超参数

- **Target Modules:** `to_q`, `to_v`, `to_k`, `to_out.0`
- **boft_block_num:** `8`
- **boft_block_size:** `0`
- **boft_n_butterfly_factor:** `1`
- **boft_dropout:** `0.1`
- **bias:** `none`（见 `data/output/boft/unet/800/adapter_config.json`）

### 3.3 训练配置

- **Resolution:** `512`
- **Max Train Steps:** `800`
- **Batch Size:** `1`
- **Learning Rate:** `3e-5`
- **Prior Loss Weight:** `1.0`
- **Checkpoint Interval:** `200`
- **Checkpoint 路径:** `data/output/boft/unet/{200,400,600,800}`

## 4. 结果与分析

### 4.1 训练收敛性

训练过程中记录了 step-level loss，并绘制为 `training_loss.png`。

![Training Loss](../training_loss.png)

观察到损失曲线整体可收敛，说明在当前参数设置下 BOFT 训练稳定、优化过程可控。

### 4.2 微调前后对比

基线（微调前）与微调后分别保存为：
- `baseline_images.png`
- `finetuned_images.png`
- 对比图 `comparison_before_after.png`

![Before vs After](../comparison_before_after.png)

定性上可见：微调后模型对 `sks dog` 的主体特征响应更强，同时仍保留一定场景组合能力。

### 4.3 多提示词泛化

在多场景提示词（如 jungle、city、street、wearing hat 等）下生成结果见：

![Multi-Prompt Results](../multi_prompt_results.png)

结果显示模型在学习主体身份后，仍可在不同语义上下文中进行组合式生成，符合 DreamBooth 个体定制任务预期。

### 4.4 参数高效性说明

Notebook 训练阶段会打印 UNet 总参数量与 BOFT 可训练参数量及占比。该机制支持直接验证“仅训练小比例参数”的 PEFT 特性。保存的输出主要为适配器权重（见 `adapter_model.safetensors`），显著小于全量模型权重规模。

## 5. 与 mini-project 要求对应关系

- **使用预训练模型：** 是（`sd2-community/stable-diffusion-2-1`）
- **使用 PEFT 方法：** 是（BOFT）
- **给出任务定义与方法说明：** 是
- **给出实验配置与复现路径：** 是（`boft.ipynb` + `environment.yml`）
- **提供结果可视化：** 是（loss、before/after、multi-prompt）
- **进行结果分析：** 是（收敛性、主体一致性、可组合性、参数效率）

## 6. 结论

基于当前仓库代码与已生成结果，本项目完成了一个可复现的 BOFT DreamBooth 微调流程。BOFT 在较低参数开销下实现了主体注入，并通过 prior preservation 保持了一定通用生成能力。实验结果与 mini-project 的“参数高效微调 + 下游任务验证”目标一致。

## 7. 参考文献

1. Qiu et al., **BOFT: Butterfly Orthogonal Fine-Tuning**, ICLR 2024.  
2. Qiu et al., **OFT: Orthogonal Finetuning for Large Models**, 2023.  
3. Hugging Face, **PEFT Documentation**: https://huggingface.co/docs/peft