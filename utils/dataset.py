"""
utils/dataset.py

PyTorch Dataset and DataLoader for the
Hybrid VAE-AttentionGAN-GMM project.
"""

import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import config


class SentinelDataset(Dataset):
    """
    Dataset for Sentinel-2 patches.
    """

    def __init__(self, npy_file):

        if not os.path.exists(npy_file):
            raise FileNotFoundError(
                f"{npy_file} not found."
            )

        self.data = np.load(npy_file).astype(np.float32)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index]

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image.permute(2, 0, 1)

        return image


def create_dataloader(
    npy_file,
    batch_size=config.BATCH_SIZE,
    shuffle=True
):
    """
    Create PyTorch DataLoader.
    """

    dataset = SentinelDataset(npy_file)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return loader


if __name__ == "__main__":

    train_loader = create_dataloader(
        config.TRAIN_DATA
    )

    print("=" * 50)

    for batch in train_loader:

        print("Batch Shape:", batch.shape)

        break

    print("=" * 50)
