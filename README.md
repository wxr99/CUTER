# CUTER: Cut out and Replay for Multi-Label Online Continual Learning

[![arXiv](https://img.shields.io/badge/ICML-2025-blue)](https://icml.cc)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **CUTER**, presented at ICML 2025.

> **Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning**
>
> Xinrui Wang, Shao-Yuan Li, Jiaqiang Zhang, Songcan Chen
>
> *International Conference on Machine Learning (ICML), 2025*

## Overview

Multi-Label Online Continual Learning (MOCL) requires models to learn continuously from endless multi-label data streams, facing catastrophic forgetting, pervasive missing labels, and uncontrollable class imbalance. Existing methods overlook a fundamental question: *which part of an image corresponds to each of its labels?*

CUTER addresses this by identifying, cropping, and replaying label-specific regions instead of whole images. This transforms multi-label replay into multiple single-label sub-tasks, providing cleaner supervision signals and finer control over class distributions. CUTER consists of three components:

- **Zero-shot localization assessment via Fiedler value**: an annotation-free metric based on spectral graph theory to evaluate pre-trained models' inherent localization capability.
- **Selective label-region matching and replay**: uses MaskCut to extract object regions, matches them to single labels via confidence filtering, and stores only high-quality label-specific patches.
- **Localization-aware feature regularization**: a nuclear norm constraint on the feature adjacency matrix, grounded in spectral perturbation theory, to prevent the deterioration of localization ability during continual learning.

## Key Results

CUTER consistently outperforms 10 baseline methods on PASCAL VOC 2007, MS-COCO 2014, and NUS-WIDE, with particularly significant gains in final (last) performance—demonstrating that precise region-level replay accumulates advantage over long-term continual learning.

## Installation

```bash
git clone https://github.com/wxr99/Cut-Replay.git
cd Cut-Replay
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.13+, torchvision, timm, numpy, scipy, scikit-learn, PIL.

## Pre-trained Models

CUTER relies on the zero-shot localization capability of pre-trained Vision Transformers. We recommend the following models, ranked by their localization quality (lower Fiedler value = better localization):

| Model | Pre-training | Link |
|---|---|---|
| ViT-S/16 (DINO v1) | Self-supervised, ImageNet-1k | [facebook/dino:main](https://github.com/facebookresearch/dino) |
| ViT-S/16 (DINO v2) | Self-supervised, LVD-142M | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| ViT-S/16 (MoCo v3) | Self-supervised, ImageNet-1k | [facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3) |
| ViT-S/16 (MAE) | Self-supervised, ImageNet-1k | [facebookresearch/mae](https://github.com/facebookresearch/mae) |
| ViT-S/16 (Supervised) | Supervised, ImageNet-21k | `timm/vit_small_patch16_224.augreg_in21k` |

**Default backbone:** ViT-S/16 with DINO v1 pre-training (ImageNet-21k). The checkpoint is automatically downloaded via `torch.hub` or `timm`.

To evaluate a custom pre-trained model's localization potential, use the Fiedler value assessment script:

```bash
python tools/eval_fiedler.py --model dino_vit_s16 --dataset voc --num_samples 500
```

## Quick Start

### Data Preparation

Datasets are downloaded automatically where possible. For NUS-WIDE (no longer publicly available in its original form), we provide a reconstruction script:

```bash
python data/prepare_nuswide.py --flickr_api_key YOUR_KEY
```

### Training

```bash
# PASCAL VOC 2007 (5 tasks, 4 classes each)
python main.py --dataset voc --num_tasks 5 --buffer_size 1000

# MS-COCO 2014 (8 tasks, 10 classes each)
python main.py --dataset coco --num_tasks 8 --buffer_size 1000

# NUS-WIDE (8 tasks)
python main.py --dataset nuswide --num_tasks 8 --buffer_size 1000
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset (voc, coco, nuswide) | voc |
| `--num_tasks` | Number of incremental tasks | 5 |
| `--buffer_size` | Memory buffer size (in number of stored regions) | 1000 |
| `--backbone` | Model backbone | vit_small_patch16_224 |
| `--pretrain` | Pre-training method (dino, mocov3, mae, supervised) | dino |
| `--num_masks` | Number of MCut iterations per image (N) | 3 |
| `--tau1` | Confidence threshold for tail classes | 0.6 |
| `--tau2` | Confidence threshold for head classes | 0.8 |
| `--alpha` | Low-rank regularization coefficient | 0.01 |
| `--lr` | Learning rate | 1e-4 |
| `--batch_size` | Training batch size | 32 |

### Running Baselines

```bash
# Reservoir Sampling (RS)
python main.py --method rs --dataset voc ...

# Partitioning Reservoir Sampling (PRS)
python main.py --method prs --dataset voc ...

# OCDM (Optimizing Class Distribution in Memory)
python main.py --method ocdm --dataset voc ...

# AGCN / AGCN++
python main.py --method agcn --dataset voc ...

# KRT (Knowledge Restore and Transfer)
python main.py --method krt --dataset voc ...
```

## Project Structure

```
CUTER/
├── main.py                 # Entry point
├── config/                 # Configuration files
├── models/
│   ├── cuter.py            # CUTER model
│   ├── backbone.py         # Feature extractors
│   └── asl_loss.py         # Asymmetric loss
├── localization/
│   ├── maskcut.py          # MaskCut for object extraction
│   ├── fiedler.py          # Fiedler value computation
│   └── lowrank.py          # Nuclear norm regularization
├── buffer/
│   ├── reservoir.py        # Class-balanced reservoir sampling
│   └── prs.py              # PRS sampling
├── utils/
│   ├── metrics.py          # mAP, CF1, OF1 evaluation
│   └── data.py             # Data loading & stream construction
├── tools/
│   └── eval_fiedler.py     # Fiedler value assessment script
└── scripts/                # Experiment scripts
```

## Citation

```bibtex
@inproceedings{wang2025cuter,
  title     = {Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning},
  author    = {Wang, Xinrui and Li, Shao-Yuan and Zhang, Jiaqiang and Chen, Songcan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

This work was supported by the National Science and Technology Major Project (2022ZD0114801), the National Natural Science Foundation of China (No. 62376126), the Funding for Outstanding Doctoral Dissertation in NUAA (BCXJ25-21), and the Major Special Basic Research Project for Aero-engines and Gas Turbines (J2019-IV-0018-0086).
