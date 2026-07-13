# Hybrid VAE-AttentionGAN-GMM for Land Use/Land Cover Classification

## Overview

This repository contains a PyTorch implementation of a Hybrid Variational Autoencoder (VAE), Attention-based Generative Adversarial Network (AttentionGAN), and Gaussian Mixture Model (GMM) framework for Land Use/Land Cover (LULC) classification using multispectral Sentinel-2 satellite imagery.

The project is designed to provide an end-to-end workflow including:

- Sentinel-2 data preprocessing
- Patch extraction
- Variational Autoencoder (VAE)
- Attention-based GAN
- Gaussian Mixture Model (GMM)
- Model training
- Performance evaluation
- Visualization
- Inference on new satellite images

---

# Repository Structure

```
Hybrid_VAE_AttentionGAN_GMM/
│
├── config.py
├── train.py
├── evaluate.py
├── inference.py
│
├── models/
│   ├── vae.py
│   ├── attention.py
│   ├── generator.py
│   ├── discriminator.py
│   └── gmm.py
│
├── utils/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
│
├── notebooks/
│
├── data/
│
├── outputs/
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

# Dataset

This project uses Sentinel-2 Level-2A multispectral satellite imagery.

Required spectral bands:

- B02 (Blue)
- B03 (Green)
- B04 (Red)
- B08 (Near Infrared)
- B11 (SWIR1)
- B12 (SWIR2)

Patch Size:

```
64 × 64 pixels
```

---

# Model Architecture

The proposed framework consists of:

1. Variational Autoencoder (VAE)
2. Attention Module (CBAM)
3. Generator Network
4. Discriminator Network
5. Gaussian Mixture Model

---

# Installation

Clone the repository:

```bash
git clone https://github.com/19022p0571/Hybrid_VAE_AttentionGAN_GMM.git
```

Move into the project directory:

```bash
cd Hybrid_VAE_AttentionGAN_GMM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

---

# Evaluation

```bash
python evaluate.py
```

---

# Inference

```bash
python inference.py
```

---

# Outputs

Training generates:

- Trained VAE
- Generator
- Discriminator
- GMM model
- Loss curves
- Accuracy plots
- Confusion matrix
- CSV reports
- Classified images

---

# Requirements

- Python 3.11+
- PyTorch
- NumPy
- Rasterio
- Scikit-learn
- OpenCV
- Matplotlib
- Pandas

---

# Citation

If you use this repository in your research, please cite the associated publication.

---

# License

This project is released under the MIT License.
