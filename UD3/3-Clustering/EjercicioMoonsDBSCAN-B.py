# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:20:48 2024

@author: MarkelP

Copilot en python Aplica el algoritmo DBSCAN al dataset moons 
Da 4 valores a eps y min_samples 
Visualiza los cuatro casos graficamente los puntos
Pon leyenda y x e y de lo que es cada cosa
 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Crear el conjunto de datos "moons"
X, _ = make_moons(n_samples=400, noise=0.05, random_state=0)

# Valores para eps y min_samples
eps_values = [0.1, 0.2, 0.3, 0.4]
min_samples_values = [5, 10, 15, 20]

# Graficar los puntos con diferentes parámetros
plt.figure(figsize=(12, 8))
for i, (eps, min_samples) in enumerate(zip(eps_values, min_samples_values)):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X)
    
    plt.subplot(2, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
    plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
