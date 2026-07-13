import torch

from models.vae import VAE

model = VAE()

x = torch.randn(4, 6, 64, 64)

reconstruction, mu, logvar = model(x)

print("Input           :", x.shape)
print("Reconstruction  :", reconstruction.shape)
print("Mu              :", mu.shape)
print("LogVar          :", logvar.shape)
