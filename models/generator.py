"""
models/generator.py

Generator network for Hybrid VAE-AttentionGAN-GMM.
"""

import torch
import torch.nn as nn

import config
from models.attention import SelfAttention


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            config.LATENT_DIM,
            256 * 8 * 8
        )

        self.net = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            SelfAttention(128),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64,
                config.NUM_CHANNELS,
                4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, 256, 8, 8)

        x = self.net(x)

        return x
