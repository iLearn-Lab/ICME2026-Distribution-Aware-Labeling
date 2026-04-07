# Beyond Quantity: Distribution-Aware Labeling for Visual Grounding

> A pipeline method for referring expression comprehension / segmentation pseudo-label generation.

## Authors

**Yichi Zhang**, **Gongwei Chen**, **Jun Zhu**, **Jia Wan**\*

\* Corresponding author


## Links

- **Paper**: `https://arxiv.org/abs/2505.24372v2`
- **Code Repository**: `https://arxiv.org/abs/2505.24372v2`


## Updates

- [12/2025] Release arXiv version
- [04/2026] Release code

     
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [TODO](#todo)
- [Citation](#citation)
- [License](#license)


## Introduction

This repository implements a pipeline for visual referring expression generation, primarily targeting the `MSCOCO` images.

The code combines multimodal large models and vision detectors to generate object descriptions, bounding boxes, and segmentation masks, then performs filtering, reassembly, and OOD-aware data generation.

This project is suitable for:

- reference expression data augmentation and distribution-aware filtering
- pseudo-label generation for image object description and localization

---

## Installation

Create and activate a Conda environment first:

```bash
conda create -n DAL python=3.11 -y
conda activate DAL
```

Install PyTorch and CUDA with Conda, then use pip for the remaining dependencies:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Models Prepare

This project currently depends on the following model directories:

- `models/Grounding-Dino-Base`
- `models/Qwen2.5-VL-7B-Instruct`
- `models/bert-base-uncased`
- `models/clip-vit-base-patch32`


Please make sure the corresponding weights and configuration files are available in these folders.

## Data Prepare

Before running the pipeline, download the MSCOCO training images to:

- `/datasets/images/mscoco/train2014`


## Usage

### 1. Generate pseudo labels under the pretrain setting

```bash
python rec_generation_with_pretrain_setting.py \
  --input_filename <input_file> \
  --output_dir <output_dir> \
  --world_size 8 \
  --gdino_path models/Grounding-Dino-Base \
  --qwen_path models/Qwen2.5-VL-3B-Instruct
```

The `<input_file>` should be a JSON file containing the paths of all images to be annotated.

### 2. Filter generated results
```bash
python filter.py \
  --input_filename <input_file> \
  --output_filename <output_dir> \
  --world_size 8
```

### 3. Convert REC pseudo-labels to RES and GRES pseudo-labels

```bash
# step 1: REC to RES
python convert_rec_to_res.py \
  --input_filename <input_file> \
  --output_filename <output_dir>

# step 2: RES to GRES
python convert_res_to_gres.py \
  --input_filename <input_file> \
  --output_filename <output_file>
```

---

## TODO

- [ ] Release the DPO code
- [ ] Release the generated data
- [ ] Release the checkpoints trained on our data

---

## Citation

If this repository corresponds to a paper, add the BibTeX entry here. Example:

```bibtex
@article{zhang2025beyond,
  title={Beyond Quantity: Distribution-Aware Labeling for Visual Grounding},
  author={Zhang, Yichi and Chen, Gongwei and Zhu, Jun and Wan, Jia},
  journal={arXiv preprint arXiv:2505.24372},
  year={2025}
}
```

---

## License

This project is released under the Apache License 2.0.
