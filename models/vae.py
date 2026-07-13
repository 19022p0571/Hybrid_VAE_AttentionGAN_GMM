"""
models/vae.py

Variational Autoencoder (VAE) for
Hybrid VAE-AttentionGAN-GMM.
"""

import torch
import torch.nn as nn

import config


class Encoder(nn.Module):
    """
    Encoder Network
    """

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(config.NUM_CHANNELS, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            128 * 8 * 8,
            config.LATENT_DIM
        )

        self.fc_logvar = nn.Linear(
            128 * 8 * 8,
            config.LATENT_DIM
        )

    def forward(self, x):

        x = self.features(x)

        x = self.flatten(x)

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder Network
    """

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            config.LATENT_DIM,
            128 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                config.NUM_CHANNELS,
                4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, 128, 8, 8)

        x = self.decoder(x)

        return x
        
class VAE(nn.Module):
    """
    Complete Variational Autoencoder
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):

        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar


import torch.nn.functional as F


def vae_loss(reconstruction, target, mu, logvar):
    """
    Computes the Variational Autoencoder loss.

    Total Loss = Reconstruction Loss + KL Divergence
    """

    reconstruction_loss = F.mse_loss(
        reconstruction,
        target,
        reduction="mean"
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss
    }
