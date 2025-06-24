# Sleep Signal Analysis with Self-Supervised Learning

This repository contains the official implementation of our paper on sleep signal analysis using self-supervised learning techniques.

## 🚀 Overview

This project implements a multi-temporal scale learning framework for sleep signal analysis using transformer-based models. The approach leverages self-supervised learning to extract meaningful representations from polysomnography (PSG) data.

### Key Features

- **Self-Supervised Learning**: Pre-training on unlabeled sleep data
- **Multiple Training Modes**: SSL, supervised learning, fine-tuning, and linear probing
- **Transformer Architecture**: MTS_LOF_revised model for temporal sequence modeling
- **Mixed Precision Training**: Efficient training with automatic mixed precision
- **SHHS Dataset Support**: Compatible with Sleep Heart Health Study data

## 📋 Requirements

### Dependencies

```bash
torch>=1.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.62.0
lifelines>=0.27.0
scipy>=1.7.0
statsmodels>=0.13.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sleep-ssl.git
cd sleep-ssl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

### SHHS Dataset

1. Download the Sleep Heart Health Study (SHHS) dataset
2. Preprocess the data using our preprocessing script:

```bash
python shhs-preprocess.py --input_path /path/to/raw/shhs --output_path /path/to/processed/data
```

3. The processed data should be organized as:
```
data/
├── shhs-all_signals-30s/
│   ├── train/
│   ├── valid/
│   └── test/
└── ...
```

## 🏃‍♂️ Usage

### Training Modes

Our framework supports multiple training paradigms:

#### 1. Self-Supervised Learning (SSL)
Pre-train the model on unlabeled data:

```bash
python main.py --train_mode ssl --data_path /path/to/data --seed 2019
```

#### 2. Supervised Learning
Train from scratch with labeled data:

```bash
python main.py --train_mode supervised --data_path /path/to/data --seed 2019
```

#### 3. Linear Probing
Freeze backbone and train only the classifier:

```bash
python main.py --train_mode linear_prob --data_path /path/to/data --load_para True --seed 2019
```

#### 4. Fine-tuning
Fine-tune the entire pre-trained model:

```bash
python main.py --train_mode finetune --data_path /path/to/data --load_para True --seed 2019
```

### Command Line Arguments

- `--train_mode`: Training mode (`ssl`, `supervised`, `linear_prob`, `finetune`)
- `--data_path`: Path to the processed dataset
- `--seed`: Random seed for reproducibility (default: 2019)
- `--num_workers`: Number of data loading threads (default: 4)
- `--load_para`: Whether to load pre-trained parameters (default: False)

## 📁 Project Structure

```
├── main.py                    # Main training script
├── model.py                   # Model definitions
├── utils.py                   # Utility functions and ResNet1D
├── shhs-preprocess.py         # Data preprocessing script
├── config_files/
│   └── SHHS1_Configs.py      # Configuration file for SHHS1 dataset
├── dataloader/
│   └── dataloader.py         # Data loading utilities
├── frs_compute/
│   └── frs.py                # FRS computation utilities
├── checkpoints/              # Model checkpoints directory
└── README.md                 # This file
```

## 🔧 Configuration

Model and training configurations are defined in `config_files/SHHS1_Configs.py`. Key parameters include:

- `embed_dim`: Embedding dimension
- `num_epoch`: Number of training epochs
- `learning_rate`: Learning rate for training
- `lr`: Learning rate for fine-tuning/linear probing
- `channel`: Signal channel to use
- `kernel_size`, `stride`: Convolution parameters

## 📈 Model Architecture

The self-supervised  model consists of:

1. **Convolutional Encoder**: 1D convolutions for local feature extraction
2. **Transformer Encoder**: Multi-head attention for temporal modeling

## 🎯 Results

Our method achieves state-of-the-art performance on sleep analysis tasks. Detailed results and comparisons are available in our paper.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please contact:
- ZhengXiao He zhengxiaohe@arizona.edu
- Project Issues: [GitHub Issues](https://github.com/yourusername/sleep-ssl/issues)

## 🙏 Acknowledgments

- Sleep Heart Health Study (SHHS) for providing the dataset
- The PyTorch team for the deep learning framework
- All contributors and collaborators

---
We only open-source part of the codes here since the paper is not published yet. After publishing, we will publish all the analysis codes here. The current code and readme documents are just temporary and need some improvements after the publishment of our article.
⭐ If you find this repository helpful, please consider giving it a star! 