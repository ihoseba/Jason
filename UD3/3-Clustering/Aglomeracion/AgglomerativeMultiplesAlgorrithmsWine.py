"""
Representa gráficamente el ajuste óptimo de los algoritmos de clustering 
- Dbscan
- KMeans
- AgglomerativeClustering

Sobre dataset Wine

"""

from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from seaborn import scatterplot
from sklearn.datasets import load_iris


import pandas as pd
# Cargar el dataset de vino tinto
df = pd.read_csv("winequality-red.csv", delimiter=',')
# Visualizar la matriz de corre
cols = df.columns

# Seleccionar características con alta correlación para visualización
# Supongamos que 'alcohol' y 'density' están altamente correlacionadas
selected_features = ['residual sugar', 'fixed acidity']
x_to_show = 0
y_to_show = 1
y = 'quality'

# Escalar los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = df[selected_features].values# scaler.fit_transform(df[selected_features])

# Dataset
y=df['quality'].values
X=X_scaled

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
    DBSCAN(eps=1.0, min_samples=7),
    KMeans(n_clusters=4),
    AgglomerativeClustering(n_clusters=4),
]

fig, axes = plt.subplots(2, 2, figsize=(30, 10))
axes_flat  = axes.flatten()

for alg, ax in zip(algorithms,axes_flat):
    alg.fit(X)
    ya=alg.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y=ya, ax=ax)
    ax.set_title(f"{alg.__class__.__name__} - {len(np.unique(ya))} clusters")

# y los datos reales
ax=axes_flat[3]
mglearn.discrete_scatter(X[:, 0], X[:, 1], y=y, ax=ax)
ax.set_title(f"Datos Reales - {len(np.unique(y))} clusters")
