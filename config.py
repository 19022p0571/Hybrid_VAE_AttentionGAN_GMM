"""
config.py

Central configuration for the Hybrid VAE-AttentionGAN-GMM project.
"""

import os
import random
import numpy as np
import torch

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Dataset Paths
# ==========================================================

DATA_DIR = "data"

TRAIN_DATA = os.path.join(DATA_DIR, "train_patches.npy")
TEST_DATA = os.path.join(DATA_DIR, "test_patches.npy")

# ==========================================================
# Model Parameters
# ==========================================================

NUM_CHANNELS = 6
PATCH_SIZE = 64
LATENT_DIM = 100
NUM_CLASSES = 5

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = "outputs"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURE_DIR,
    CSV_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# ==========================================================
# Model Files
# ==========================================================

VAE_MODEL = os.path.join(CHECKPOINT_DIR, "vae_best.pth")
GENERATOR_MODEL = os.path.join(CHECKPOINT_DIR, "generator_best.pth")
DISCRIMINATOR_MODEL = os.path.join(CHECKPOINT_DIR, "discriminator_best.pth")
GMM_MODEL = os.path.join(CHECKPOINT_DIR, "gmm.pkl")

# ==========================================================
# History Files
# ==========================================================

TRAINING_HISTORY = os.path.join(CSV_DIR, "training_history.csv")

# ==========================================================
# Display Configuration
# ==========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Hybrid VAE-AttentionGAN-GMM Configuration")
    print("=" * 50)
    print(f"Device          : {DEVICE}")
    print(f"Seed            : {SEED}")
    print(f"Patch Size      : {PATCH_SIZE}")
    print(f"Latent Dim      : {LATENT_DIM}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Learning Rate   : {LEARNING_RATE}")
    print("=" * 50)
