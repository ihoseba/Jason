import mglearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#Obtenemos el conjunto de datos
iris=load_iris()


# Escoge 3 columnas del dataset de iris
# Realiza las agrupaciones de clustering con el algoritmo que mejor creas que encaja para este caso
# Representa las 3 columnas gráficamente (en 3d) y representa cada cluster on un color distinto

"""
Copilot, haz codigo python que recoja 3 columnas del dataset de iris
Realiza las agrupaciones de clustering con varios algoritmos 
Representa las 3 columnas gráficamente (en 3d) cada uno de ellos 
y representa cada cluster on un color distinto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data[:, :3]  # Seleccionar las primeras tres columnas (características)

# Aplicar K-means
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)

# Aplicar Clustering Jerárquico Aglomerativo
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Crear gráficos 3D
fig = plt.figure(figsize=(12, 6))

# K-means
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans_labels, cmap='viridis')
ax1.set_title('K-means Clustering')

# Clustering Jerárquico Aglomerativo
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=agg_labels, cmap='viridis')
ax2.set_title('Clustering Jerárquico Aglomerativo')

# DBSCAN
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=dbscan_labels, cmap='viridis')
ax3.set_title('DBSCAN Clustering')

plt.tight_layout()
plt.show()
