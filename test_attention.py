import torch

from models.attention import SelfAttention

x = torch.randn(2, 64, 32, 32)

attention = SelfAttention(64)

output = attention(x)

print("Input :", x.shape)
print("Output:", output.shape)
