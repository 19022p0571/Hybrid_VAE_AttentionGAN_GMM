import torch

from models.vae import VAE, vae_loss

model = VAE()

x = torch.randn(4, 6, 64, 64)

reconstruction, mu, logvar = model(x)

loss = vae_loss(
    reconstruction,
    x,
    mu,
    logvar
)

print("Total Loss :", loss["total_loss"].item())
print("Reconstruction Loss :", loss["reconstruction_loss"].item())
print("KL Loss :", loss["kl_loss"].item())
