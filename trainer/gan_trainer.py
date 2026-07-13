"""
trainer/gan_trainer.py

Training utilities for Generator and Discriminator.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import config


class GANTrainer:

    def __init__(self, generator, discriminator, train_loader):

        self.generator = generator.to(config.DEVICE)
        self.discriminator = discriminator.to(config.DEVICE)

        self.train_loader = train_loader

        self.criterion = nn.BCELoss()

        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999)
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999)
        )

    def train_epoch(self):

        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for real_images in self.train_loader:

            real_images = real_images.to(config.DEVICE)

            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size,
                1,
                device=config.DEVICE
            )

            fake_labels = torch.zeros(
                batch_size,
                1,
                device=config.DEVICE
            )

            # ---------------------
            # Train Discriminator
            # ---------------------

            self.optimizer_d.zero_grad()

            real_output = self.discriminator(real_images)

            d_loss_real = self.criterion(
                real_output,
                real_labels
            )

            z = torch.randn(
                batch_size,
                config.LATENT_DIM,
                device=config.DEVICE
            )

            fake_images = self.generator(z)

            fake_output = self.discriminator(
                fake_images.detach()
            )

            d_loss_fake = self.criterion(
                fake_output,
                fake_labels
            )

            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()

            self.optimizer_d.step()

            # ---------------------
            # Train Generator
            # ---------------------

            self.optimizer_g.zero_grad()

            output = self.discriminator(fake_images)

            g_loss = self.criterion(
                output,
                real_labels
            )

            g_loss.backward()

            self.optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        n = len(self.train_loader)

        return {
            "generator_loss": total_g_loss / n,
            "discriminator_loss": total_d_loss / n
        }
