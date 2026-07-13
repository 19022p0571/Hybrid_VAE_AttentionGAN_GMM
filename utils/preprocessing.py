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
