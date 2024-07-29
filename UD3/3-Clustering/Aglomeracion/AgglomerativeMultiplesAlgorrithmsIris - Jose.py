"""
Representa gráficamente el ajuste óptimo de los algoritmos de clustering 
- Dbscan
- KMeans
- AgglomerativeClustering

"""

from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from seaborn import scatterplot
from sklearn.datasets import load_iris

# Dataset
iris = load_iris()
X = iris.data[:, [0,2]]  # Seleccionar las primeras columnas (características)
y = iris.target

#Obtenemos el array de enlaces entre instancias
array_enlaces=ward(X)
#Instanciamos el dendograma

dendrogram(array_enlaces)

# Plotear dataset
plt.figure(figsize=(10, 7))
scatterplot(x=X[:, 0], y=X[:, 1],c=y)
plt.title("Datos reales con 3 clusters")
plt.show()

# Clusters optimos son 3

from sklearn.cluster import KMeans, DBSCAN
import mglearn

algorithms = [
    DBSCAN(eps=.5, min_samples=7),
    KMeans(n_clusters=3),
    AgglomerativeClustering(n_clusters=3),
]

fig, axes = plt.subplots(2, 2, figsize=(30, 10))
axes_flat  = axes.flatten()

for alg, ax in zip(algorithms,axes_flat):
    alg.fit(X)
    ya=alg.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y=ya, ax=ax)
    ax.set_title(f"{alg.__class__.__name__} - {len(np.unique(y))} clusters")

# y los datos reales
ax=axes_flat[3]
mglearn.discrete_scatter(X[:, 0], X[:, 1], y=y, ax=ax)
ax.set_title("Datos Reales con 5 clusters")
