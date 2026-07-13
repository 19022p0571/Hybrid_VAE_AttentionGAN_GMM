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




