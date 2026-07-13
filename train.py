"""
train.py

Main training script for Hybrid VAE-AttentionGAN-GMM.
"""

from utils.dataset import create_dataloader
from models.vae import VAE
from trainer.vae_trainer import VAETrainer
from models.generator import Generator
from models.discriminator import Discriminator
from trainer.gan_trainer import GANTrainer
import os

import config


def main():

    print("=" * 60)
    print("Hybrid VAE-AttentionGAN-GMM")
    print("=" * 60)

    print("Loading dataset...")

    train_loader = create_dataloader(config.TRAIN_DATA)

    print("Dataset loaded.")

    print("Initializing VAE...")

    vae = VAE()

    trainer = VAETrainer(
        vae,
        train_loader
    )

    history = trainer.train(
        epochs=config.NUM_EPOCHS
    )

    os.makedirs("checkpoints", exist_ok=True)

    trainer.save_model("checkpoints/vae_model.pth")
    print("\nInitializing GAN...")

    generator = Generator()

    discriminator = Discriminator()

    gan_trainer = GANTrainer(
    generator,
    discriminator,
    train_loader
    )

    gan_history = gan_trainer
    print("Training completed successfully.")

    return history


if __name__ == "__main__":
    main()
