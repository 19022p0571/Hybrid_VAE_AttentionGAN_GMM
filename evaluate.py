import torch
import torch.nn.functional as F

import config

from utils.dataset import create_dataloader
from models.vae import VAE
from models.generator import Generator
from models.discriminator import Discriminator


def evaluate():

    device = config.DEVICE

    print("=" * 60)
    print("Hybrid VAE-AttentionGAN-GMM Evaluation")
    print("=" * 60)

    # Load test dataset
    test_loader = create_dataloader(
        config.TEST_DATA,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Load models
    vae = VAE().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    vae.load_state_dict(
        torch.load("checkpoints/vae_model.pth",
                   map_location=device)
    )

    generator.load_state_dict(
        torch.load("checkpoints/generator.pth",
                   map_location=device)
    )

    discriminator.load_state_dict(
        torch.load("checkpoints/discriminator.pth",
                   map_location=device)
    )

    vae.eval()
    generator.eval()
    discriminator.eval()

    reconstruction_loss = 0.0
    discriminator_score = 0.0

    with torch.no_grad():

        for images in test_loader:

            images = images.to(device)

            reconstructed, mu, logvar = vae(images)

            reconstruction_loss += F.mse_loss(
                reconstructed,
                images,
                reduction="mean"
            ).item()

            prediction = discriminator(reconstructed)

            discriminator_score += prediction.mean().item()

    reconstruction_loss /= len(test_loader)
    discriminator_score /= len(test_loader)

    print("\nEvaluation Results")
    print("-" * 40)
    print(f"Average Reconstruction Loss : {reconstruction_loss:.6f}")
    print(f"Average Discriminator Score : {discriminator_score:.6f}")
    print("-" * 40)
    import os

    os.makedirs("outputs", exist_ok=True)

    results = f"""
    ========================================
    Hybrid VAE-AttentionGAN-GMM Evaluation
    ========================================

    Average Reconstruction Loss : {reconstruction_loss:.6f}
    Average Discriminator Score : {discriminator_score:.6f}
    """

    print(results)

    with open("outputs/evaluation_results.txt", "w") as f:
    	f.write(results)

    print("Results saved to outputs/evaluation_results.txt")
    import csv

    with open("outputs/metrics.csv", "w", newline="") as csvfile:
    	writer = csv.writer(csvfile)

    writer.writerow(["Metric", "Value"])
    writer.writerow(["Average Reconstruction Loss", reconstruction_loss])
    writer.writerow(["Average Discriminator Score", discriminator_score])

    print("Metrics saved to outputs/metrics.csv")


if __name__ == "__main__":
    evaluate()
