# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:37:45 2024

@author: joseangelperez

Copilot en python carga el dataset 
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
Aplica castering para comprobar correlaciones entre datos 
Visualización de resultados
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = pd.read_csv("Mall_Customers.csv")

# Seleccionar las columnas relevantes para el clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)", "Age"]]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Aplicar Clustering Jerárquico Aglomerativo
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X_scaled)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Crear gráficos 3D
fig = plt.figure(figsize=(18, 6))

# K-means
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=kmeans_labels, cmap='viridis')
ax1.set_title('K-means Clustering')
ax1.set_xlabel('Annual Income (scaled)')
ax1.set_ylabel('Spending Score (scaled)')
ax1.set_zlabel('Age (scaled)')

# Clustering Jerárquico Aglomerativo
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=agg_labels, cmap='viridis')
ax2.set_title('Clustering Jerárquico Aglomerativo')
ax2.set_xlabel('Annual Income (scaled)')
ax2.set_ylabel('Spending Score (scaled)')
ax2.set_zlabel('Age (scaled)')

# DBSCAN
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=dbscan_labels, cmap='viridis')
ax3.set_title('DBSCAN Clustering')
ax3.set_xlabel('Annual Income (scaled)')
ax3.set_ylabel('Spending Score (scaled)')
ax3.set_zlabel('Age (scaled)')

plt.tight_layout()
plt.show()

