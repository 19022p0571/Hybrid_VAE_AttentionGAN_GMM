import torch

from models.discriminator import Discriminator

model = Discriminator()

x = torch.randn(4, 6, 64, 64)

output = model(x)

print("Input :", x.shape)
print("Output:", output.shape)
