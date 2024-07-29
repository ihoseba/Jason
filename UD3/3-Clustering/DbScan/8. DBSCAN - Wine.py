# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:20:48 2024

@author: MarkelP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset de vino tinto
df = pd.read_csv("winequality-red.csv", delimiter=',')

import matplotlib.pyplot as plt
import seaborn as sns

#sns.pairplot(df, vars=df.columns, hue="quality", plot_kws={'alpha':0.6},corner=True)
#plt.show()


# Visualizar la matriz de corre
cols = df.columns

# Seleccionar características con alta correlación para visualización
# Supongamos que 'alcohol' y 'density' están altamente correlacionadas
selected_features = ['residual sugar', 'fixed acidity']
x_to_show = 0
y_to_show = 1
y = 'quality'

# Escalar los datos
scaler = StandardScaler()
X_scaled = df[selected_features].values# scaler.fit_transform(df[selected_features])

# Aplicar DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan.fit(X_scaled)

# Visualizar los resultados
labels = dbscan.labels_

plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, x_to_show], X_scaled[:, y_to_show], c=labels, cmap='Paired')
plt.xlabel(selected_features[0] + " (scaled)")
plt.ylabel(selected_features[1] + " (scaled)")
plt.title("DBSCAN Clustering on Wine Dataset")
plt.show()


# Imprimir etiquetas
print(labels)
