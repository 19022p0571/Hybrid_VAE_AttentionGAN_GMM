"""
models/gmm.py

Gaussian Mixture Model classifier for
Hybrid VAE-AttentionGAN-GMM.
"""

import joblib
from sklearn.mixture import GaussianMixture


class GMMClassifier:
    """
    Wrapper around scikit-learn's GaussianMixture.
    """

    def __init__(self, n_components=5, random_state=42):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state
        )

    def fit(self, features):
        self.model.fit(features)

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
