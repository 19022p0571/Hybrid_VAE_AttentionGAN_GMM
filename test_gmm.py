import numpy as np

from models.gmm import GMMClassifier

features = np.random.randn(100, 100)

gmm = GMMClassifier(n_components=5)

gmm.fit(features)

labels = gmm.predict(features)

print("Feature Shape :", features.shape)
print("Label Shape   :", labels.shape)
print("Unique Labels :", np.unique(labels))
