"""
trainer/vae_trainer.py

Training utilities for the VAE.
"""

import torch
import torch.optim as optim

import config
from models.vae import vae_loss


class VAETrainer:

    def __init__(self, model, train_loader):

        self.model = model.to(config.DEVICE)

        self.train_loader = train_loader

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        total_reconstruction = 0.0

        total_kl = 0.0

        for images in self.train_loader:

            images = images.to(config.DEVICE)

            self.optimizer.zero_grad()

            reconstruction, mu, logvar = self.model(images)

            loss = vae_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            loss["total_loss"].backward()

            self.optimizer.step()

            total_loss += loss["total_loss"].item()

            total_reconstruction += loss["reconstruction_loss"].item()

            total_kl += loss["kl_loss"].item()

        n = len(self.train_loader)

        return {
            "loss": total_loss / n,
            "reconstruction": total_reconstruction / n,
            "kl": total_kl / n
        }


    def train(self, epochs):

        history = []

        print("=" * 60)
        print("Starting VAE Training")
        print("=" * 60)

        for epoch in range(epochs):

            metrics = self.train_epoch()

            history.append(metrics)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {metrics['loss']:.6f} | "
                f"Recon: {metrics['reconstruction']:.6f} | "
                f"KL: {metrics['kl']:.6f}"
            )

        print("=" * 60)
        print("Training completed.")
        print("=" * 60)

        return history

    def save_model(self, filepath):

        torch.save(self.model.state_dict(), filepath)

        print(f"Model saved to: {filepath}")
