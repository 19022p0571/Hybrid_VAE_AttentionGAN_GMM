"""
train.py

Main training script for Hybrid VAE-AttentionGAN-GMM.
"""

import os

import config

from utils.dataset import create_dataloader

from models.vae import VAE
from models.generator import Generator
from models.discriminator import Discriminator

from trainer.vae_trainer import VAETrainer
from trainer.gan_trainer import GANTrainer


def main():

    print("=" * 60)
    print("Hybrid VAE-AttentionGAN-GMM")
    print("=" * 60)

    # ----------------------------------------
    # Load Dataset
    # ----------------------------------------

    print("Loading dataset...")

    train_loader = create_dataloader(config.TRAIN_DATA)

    print("Dataset loaded.")

    # ----------------------------------------
    # Create Checkpoint Folder
    # ----------------------------------------

    os.makedirs("checkpoints", exist_ok=True)

    # ----------------------------------------
    # Train VAE
    # ----------------------------------------

    print("\nInitializing VAE...")

    vae = VAE()

    vae_trainer = VAETrainer(
        vae,
        train_loader
    )

    vae_history = vae_trainer.train(
        epochs=config.NUM_EPOCHS
    )

    vae_trainer.save_model(
        "checkpoints/vae_model.pth"
    )

    print("VAE training completed successfully.")

    # ----------------------------------------
    # Train GAN
    # ----------------------------------------

    print("\nInitializing GAN...")

    generator = Generator()

    discriminator = Discriminator()

    gan_trainer = GANTrainer(
        generator,
        discriminator,
        train_loader
    )

    gan_history = gan_trainer.train(
        epochs=config.NUM_EPOCHS
    )

    gan_trainer.save_generator(
        "checkpoints/generator.pth"
    )

    gan_trainer.save_discriminator(
        "checkpoints/discriminator.pth"
    )

    print("GAN training completed successfully.")

    # ----------------------------------------
    # Finished
    # ----------------------------------------

    print("\n" + "=" * 60)
    print("Hybrid VAE-AttentionGAN-GMM Training Completed")
    print("=" * 60)

    return {
        "vae": vae_history,
        "gan": gan_history
    }


if __name__ == "__main__":
    main()
