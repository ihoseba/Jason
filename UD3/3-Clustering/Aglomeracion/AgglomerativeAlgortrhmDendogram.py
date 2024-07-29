from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward
import numpy as np
import matplotlib.pyplot as plt
from seaborn import scatterplot

# Dataset
centros_blob = np.array([[1.5,2.4],[0.5,2.3],[-0.5,2],[-1,3],[-1.5,2.6]])
blob_std = np.array([0.3, 0.25, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=800, centers=centros_blob,
cluster_std=blob_std,random_state=20)

#Obtenemos el array de enlaces entre instancias
array_enlaces=ward(X)
#Instanciamos el dendograma
dendrogram(array_enlaces)

# Plotear dataset
plt.figure(figsize=(10, 7))
scatterplot(x=X[:, 0], y=X[:, 1],c=y)
plt.show()

# Clusters optimos son 5

from sklearn.cluster import KMeans, DBSCAN
import mglearn

algorithms = [
    DBSCAN(eps=.15, min_samples=7),
    KMeans(n_clusters=5),
]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes_flat  = axes.flatten()
for alg,ax in zip(algorithms,axes_flat):
    alg.fit(X)
    y=alg.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y=y, ax=ax)

axes.show()
