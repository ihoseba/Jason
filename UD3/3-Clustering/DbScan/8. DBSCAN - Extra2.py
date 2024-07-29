import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

from seaborn import pairplot
from seaborn import scatterplot

datasets = [
    (noisy_circles,0.2,5),
    (noisy_moons,0.2,5),
    (blobs,0.4,7),
    (aniso,0.4,5),
    (varied,1.0,17),
    (no_structure,0.03,7)
]

eps=0.2
min_samples=5
for ds,eps,min_smples in datasets:
    X = ds[0]
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X)

    

    plt.figure(figsize=(10, 7))
    scatterplot(x=X[:, 0], y=X[:, 1],c=clusters)
    plt.show()