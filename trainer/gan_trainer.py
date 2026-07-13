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
