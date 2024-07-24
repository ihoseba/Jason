# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:50:07 2024

@author: joseangelperez

Copilot, genera un codigo python que partiendo de un csv llamado winequality-red.csv, y
trabajando unicamente con alcohol y pH 
e ignorando la clase resultado.
Agrupa según el algoritmo de las K-medias barriendo casos desde 2 hasta 8 grupos.
Representa la gráfica de la inercia y del índice de siluetas para cada uno de los casos.
Obtén también el diagrama de silueta considerando los resultados previos. 

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Cargar el archivo CSV
data = pd.read_csv("winequality-red.csv")

# Seleccionar las columnas "alcohol" y "pH"
selected_columns = ["alcohol", "sulphates"]
X = data[selected_columns]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular la inercia para diferentes valores de k (2 a 8)
inertia_values = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Graficar la inercia
plt.figure(figsize=(8, 6))
plt.plot(range(2, 9), inertia_values, marker="o")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Inercia")
plt.title("Gráfica de Inercia")
plt.show()

# Calcular el índice de siluetas para cada k
silhouette_scores = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Graficar el índice de siluetas
plt.figure(figsize=(8, 6))
plt.plot(range(2, 9), silhouette_scores, marker="o")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Índice de Siluetas")
plt.title("Gráfica de Índice de Siluetas")
plt.show()

# Obtener el diagrama de silueta para k=3 (por ejemplo)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
silhouette_values = silhouette_samples(X_scaled, labels)

# Graficar el diagrama de silueta
plt.figure(figsize=(8, 6))
y_lower = 10
for i in range(3):
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(range(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.xlabel("Coeficiente de Silueta")
plt.ylabel("Etiqueta del Grupo")
plt.title("Diagrama de Silueta para k=3")
plt.show()

