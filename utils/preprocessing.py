"""
utils/preprocessing.py

Sentinel-2 preprocessing utilities for the
Hybrid VAE-AttentionGAN-GMM project.

Author: P S V Durga Gayatri
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split

import config


class Sentinel2Preprocessor:
    """
    Sentinel-2 Level-2A preprocessing.
    """

    def __init__(self, safe_folder):

        self.safe_folder = safe_folder

        self.granule = None

        self.img_data = None

        self.r10 = None

        self.r20 = None

        self.image = None

        self.band_files = {}

    def locate_folders(self):
        """
        Locate GRANULE, IMG_DATA,
        R10m and R20m folders.
        """

        granules = glob.glob(
            os.path.join(
                self.safe_folder,
                "GRANULE",
                "*"
            )
        )

        if len(granules) == 0:
            raise FileNotFoundError(
                "GRANULE folder not found."
            )

        self.granule = granules[0]

        self.img_data = os.path.join(
            self.granule,
            "IMG_DATA"
        )

        self.r10 = os.path.join(
            self.img_data,
            "R10m"
        )

        self.r20 = os.path.join(
            self.img_data,
            "R20m"
        )

        print("R10m :", self.r10)
        print("R20m :", self.r20)

    def find_band(self, band_name, resolution):
        """
        Find Sentinel-2 band.
        """

        folder = self.r10 if resolution == "10m" else self.r20

        files = glob.glob(
            os.path.join(
                folder,
                f"*{band_name}_{resolution}.jp2"
            )
        )

        if len(files) == 0:

            raise FileNotFoundError(
                f"{band_name} not found."
            )

        return files[0]

    def locate_band_files(self):
        """
        Locate all required bands.
        """

        self.band_files["B02"] = self.find_band(
            "B02",
            "10m"
        )

        self.band_files["B03"] = self.find_band(
            "B03",
            "10m"
        )

        self.band_files["B04"] = self.find_band(
            "B04",
            "10m"
        )

        self.band_files["B08"] = self.find_band(
            "B08",
            "10m"
        )

        self.band_files["B11"] = self.find_band(
            "B11",
            "20m"
        )

        self.band_files["B12"] = self.find_band(
            "B12",
            "20m"
        )

        print("\nDetected Bands")

        for band, file in self.band_files.items():

            print(f"{band} -> {os.path.basename(file)}")
    def read_band(self, filename, out_shape=None):
        """
        Read a Sentinel-2 band.
        If out_shape is provided, the image is resampled.
        """

        with rasterio.open(filename) as src:

            if out_shape is None:

                image = src.read(1)

            else:

                image = src.read(
                    1,
                    out_shape=out_shape,
                    resampling=Resampling.bilinear
                )

        return image

    def load_bands(self):
        """
        Load and stack the six Sentinel-2 bands.
        """

        print("\nLoading Sentinel-2 Bands...")

        blue = self.read_band(
            self.band_files["B02"]
        )

        green = self.read_band(
            self.band_files["B03"]
        )

        red = self.read_band(
            self.band_files["B04"]
        )

        nir = self.read_band(
            self.band_files["B08"]
        )

        target_shape = blue.shape

        swir1 = self.read_band(
            self.band_files["B11"],
            out_shape=target_shape
        )

        swir2 = self.read_band(
            self.band_files["B12"],
            out_shape=target_shape
        )

        self.image = np.stack(
            [
                blue,
                green,
                red,
                nir,
                swir1,
                swir2
            ],
            axis=-1
        ).astype(np.float32)

        print("Stacked Image Shape:", self.image.shape)

        return self.image

    def normalize(self):
        """
        Normalize Sentinel-2 reflectance values.
        """

        if self.image is None:
            raise ValueError("Image not loaded.")

        self.image = self.image / 10000.0
        self.image = np.clip(self.image, 0.0, 1.0)

        print(
            "Normalized Range:",
            self.image.min(),
            self.image.max()
        )

        return self.image

    def show_rgb(self):
        """
        Display RGB image.
        """

        if self.image is None:
            raise ValueError("Image not loaded.")

        rgb = self.image[:, :, [2, 1, 0]]

        plt.figure(figsize=(8, 8))

        plt.imshow(rgb)

        plt.title("Sentinel-2 RGB")

        plt.axis("off")

        plt.show()
    def extract_patches(self, patch_size=config.PATCH_SIZE, stride=None):
        """
        Extract non-overlapping or overlapping patches.
        """

        if self.image is None:
            raise ValueError("Image not loaded.")

        if stride is None:
            stride = patch_size

        patches = []
        metadata = []

        height, width, _ = self.image.shape

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):

                patch = self.image[
                    y:y + patch_size,
                    x:x + patch_size,
                    :
                ]

                patches.append(patch)

                metadata.append({
                    "x": x,
                    "y": y
                })

        patches = np.asarray(patches, dtype=np.float32)

        metadata = pd.DataFrame(metadata)

        print(f"Extracted {len(patches)} patches.")
        print("Patch Shape:", patches.shape)

        return patches, metadata

    def split_dataset(
        self,
        patches,
        metadata,
        test_size=0.2,
        random_state=config.SEED
    ):
        """
        Split patches into training and testing sets.
        """

        train_patches, test_patches, train_meta, test_meta = train_test_split(
            patches,
            metadata,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        return (
            train_patches,
            test_patches,
            train_meta,
            test_meta
        )

    def save_dataset(
        self,
        train_patches,
        test_patches,
        train_meta,
        test_meta,
        output_dir=config.DATA_DIR
    ):
        """
        Save dataset to disk.
        """

        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "train_patches.npy"),
            train_patches
        )

        np.save(
            os.path.join(output_dir, "test_patches.npy"),
            test_patches
        )

        train_meta.to_csv(
            os.path.join(output_dir, "train_metadata.csv"),
            index=False
        )

        test_meta.to_csv(
            os.path.join(output_dir, "test_metadata.csv"),
            index=False
        )

        print("\nDataset saved successfully.")
        print(f"Training patches: {len(train_patches)}")
        print(f"Testing patches : {len(test_patches)}")
      def run_pipeline(
        self,
        patch_size=config.PATCH_SIZE,
        stride=None,
        test_size=0.2
    ):
        """
        Complete Sentinel-2 preprocessing pipeline.
        """

        print("=" * 60)
        print("Hybrid VAE-AttentionGAN-GMM")
        print("Sentinel-2 Preprocessing Pipeline")
        print("=" * 60)

        # Step 1
        print("\n[1/7] Locating folders...")
        self.locate_folders()

        # Step 2
        print("\n[2/7] Locating band files...")
        self.locate_band_files()

        # Step 3
        print("\n[3/7] Loading bands...")
        self.load_bands()

        # Step 4
        print("\n[4/7] Normalizing image...")
        self.normalize()

        # Step 5
        print("\n[5/7] Extracting patches...")
        patches, metadata = self.extract_patches(
            patch_size=patch_size,
            stride=stride
        )

        # Step 6
        print("\n[6/7] Splitting dataset...")
        (
            train_patches,
            test_patches,
            train_meta,
            test_meta
        ) = self.split_dataset(
            patches,
            metadata,
            test_size=test_size
        )

        # Step 7
        print("\n[7/7] Saving dataset...")
        self.save_dataset(
            train_patches,
            test_patches,
            train_meta,
            test_meta
        )

        print("\n" + "=" * 60)
        print("Preprocessing completed successfully.")
        print("=" * 60)

        return {
            "train_patches": train_patches,
            "test_patches": test_patches,
            "train_metadata": train_meta,
            "test_metadata": test_meta
        }



