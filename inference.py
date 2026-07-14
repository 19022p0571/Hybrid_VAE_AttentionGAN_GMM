import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import config

from models.vae import VAE


def inference():

    device = config.DEVICE

    print("=" * 60)
    print("Hybrid VAE-AttentionGAN-GMM Inference")
    print("=" * 60)

    # Load trained VAE
    model = VAE().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/vae_model.pth",
            map_location=device
        )
    )

    model.eval()

    # Load test dataset
    data = np.load(config.TEST_DATA)

    sample = data[0]

    image = torch.tensor(
        sample,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        reconstruction, _, _ = model(image)

    original = image.squeeze().permute(1, 2, 0).cpu().numpy()

    reconstructed = (
        reconstruction.squeeze()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(original[:, :, :3])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(reconstructed[:, :, :3])
    plt.title("Reconstructed")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig("outputs/reconstruction.png")

    plt.show()

    print("\nSaved to outputs/reconstruction.png")


if __name__ == "__main__":
    inference()
