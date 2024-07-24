# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:03:59 2024

@author: joseangelperez

Copilot genera codigo en python para aplicar el algoritmo DBSCAN a datos extraidos 
de un fichero csv a cargar llamado winequality-red.csv
Da 4 valores a eps y min_samples 
Visualiza los cuatro casos graficamente los puntos
Pon leyenda y x e y de lo que es cada cosa

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Cargar el archivo CSV
data = pd.read_csv("winequality-red.csv")

parametros=["fixed acidity","volatile acidity","citric acid","residual sugar",
            "chlorides","free sulfur dioxide","total sulfur dioxide",
            "density","pH","sulphates","alcohol","quality"]
par1="free sulfur dioxide"
par2="volatile acidity"

# Extraer las características relevantes (por ejemplo, 4 columnas)
#features = data[["alcohol", "total sulfur dioxide", "pH", "residual sugar"]]
features = data[[par1,par2]]

# Hiperparámetros para DBSCAN
eps_values = [0.1, 0.5, 1.0, 2.0]  # Valores de eps
min_samples_values = [5, 10, 20, 30]  # Valores de min_samples

# Crear subplots para los gráficos
fig, axs = plt.subplots(len(eps_values), len(min_samples_values), figsize=(12, 12))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        # Crear el modelo DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(features)

        # Etiquetar los puntos
        labels = dbscan.labels_

        # Graficar los puntos
        axs[i, j].scatter(features[par1], features[par2],
                          s=1, c=labels, cmap="viridis")
        axs[i, j].set_title(f"eps={eps}, min_samples={min_samples}")
        axs[i, j].set_xlabel(par1)
        axs[i, j].set_ylabel(par2)

# Añadir leyenda
plt.legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()


