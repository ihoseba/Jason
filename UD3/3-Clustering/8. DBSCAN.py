# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:20:48 2024

@author: MarkelP


"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
import mglearn
import numpy as np

centros_blob = np.array([[-12, -5], [-6, 2], [3, 0]])
blob_std = np.array([1, 2.5, 0.5])

X, y = make_blobs(
    centers=centros_blob,
    cluster_std=blob_std,
    n_samples=200,
    random_state=20
)

mglearn.discrete_scatter(X[: ,0], X[: ,1])


dbsan = DBSCAN(eps=3.2, min_samples=15)
dbsan.fit(X)

mglearn.discrete_scatter(X[: ,0], X[: ,1], dbsan.labels_, markers='o')
print(dbsan.labels_)