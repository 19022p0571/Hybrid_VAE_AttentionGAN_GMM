import torch

from models.vae import Encoder, Decoder

encoder = Encoder()

decoder = Decoder()

x = torch.randn(2, 6, 64, 64)

mu, logvar = encoder(x)

print("Mu:", mu.shape)

print("LogVar:", logvar.shape)

z = torch.randn(2, 100)

output = decoder(z)

print("Output:", output.shape)
