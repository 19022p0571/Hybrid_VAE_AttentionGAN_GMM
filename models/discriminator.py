"""
models/discriminator.py

Discriminator for Hybrid VAE-AttentionGAN-GMM.
"""

import torch.nn as nn

import config
from models.attention import SelfAttention


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                config.NUM_CHANNELS,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(64),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 1),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
