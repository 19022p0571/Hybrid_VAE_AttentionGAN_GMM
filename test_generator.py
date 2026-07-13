import torch

from models.generator import Generator
import config

generator = Generator()

z = torch.randn(4, config.LATENT_DIM)

fake = generator(z)

print("Latent Vector :", z.shape)
print("Generated Image:", fake.shape)
