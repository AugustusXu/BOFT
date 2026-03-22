# BOFT DreamBooth Mini-Project

This project is based on `boft.ipynb` and uses **BOFT (Butterfly Orthogonal Fine-Tuning)** to finetune a pretrained Stable Diffusion model for subject-driven image generation (DreamBooth).

## 1. Project Goals (Aligned with the mini-project requirements)

- Select a downstream task: DreamBooth subject personalization
- Use a pretrained foundation model: `sd2-community/stable-diffusion-2-1`
- Apply a parameter-efficient finetuning method: BOFT (PEFT)
- Provide before/after visual comparisons and result analysis

## 2. Method Overview

BOFT injects trainable orthogonal matrices with butterfly factorization into UNet attention layers. Only adapter parameters are trained, while pretrained model weights remain frozen.

Key properties:

- Parameter-efficient: only a small fraction of parameters are trainable (the notebook prints the trainable ratio)
- Orthogonal constraint: helps preserve the representation geometry of the pretrained model
- Full-rank updates: stronger expressiveness than low-rank additive updates

References:
- [BOFT Paper (ICLR 2024)](https://arxiv.org/abs/2311.06243)
- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 3. Current Code and Directory Layout (Aligned with this repository)

```text
BOFT/
├── boft.ipynb
├── environment.yml
├── README.md
├── README.en.md
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
    ├── report.md
    └── report.en.md
```

## 4. Experiment Settings (From `boft.ipynb`)

### Model and Task

- Base model: `sd2-community/stable-diffusion-2-1`
- Unique token: `sks`
- Instance prompt: `a photo of sks dog`
- Class prompt: `a photo of dog`

### Training Hyperparameters

- `RESOLUTION=512`
- `MAX_TRAIN_STEPS=800`
- `LEARNING_RATE=3e-5`
- `TRAIN_BATCH_SIZE=1`
- `NUM_CLASS_IMAGES=100`
- `CHECKPOINT_STEPS=200`
- `PRIOR_LOSS_WEIGHT=1.0`

### BOFT Hyperparameters

- `boft_block_num=8`
- `boft_block_size=0`
- `boft_n_butterfly_factor=1`
- `boft_dropout=0.1`
- `target_modules=[to_q, to_v, to_k, to_out.0]`
- `bias=none` (see `data/output/boft/unet/800/adapter_config.json`)

## 5. How to Run

### 5.1 Create the Environment

```bash
conda env create -f environment.yml
conda activate aist
```

### 5.2 Launch the Notebook

```bash
jupyter notebook boft.ipynb
```

Run cells in order to complete:
1. Data preparation and visualization
2. Baseline generation before finetuning
3. BOFT finetuning (with checkpoints)
4. Training loss visualization
5. Post-finetuning generation and qualitative comparison
6. Multi-prompt generalization generation

## 6. Result Files

- Baseline images: `baseline_images.png`
- Finetuned images: `finetuned_images.png`
- Before/after comparison: `comparison_before_after.png`
- Multi-prompt results: `multi_prompt_results.png`
- Training loss curve: `training_loss.png`
- Adapter checkpoints: `data/output/boft/unet/{200,400,600,800}`

## 7. Notes

- This repository is notebook-centered and does not include a standalone `train_dreambooth.py`.
- `report/report.md` has been synced with the actual configuration and artifacts in this repository.
