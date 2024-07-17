# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:10:31 2024

@author: joseangelperez

Utiliza el dataset Wine disponible en sklearn.datasets. Clasifica el dataset utilizando:
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
Determina la exactitud alcanzada en cada caso y comenta los resultados. Representa gráficamente las fronteras de decisión para cada método)
    
Copilot genera un codigo en python que utiliza el dataset Wine disponible en sklearn.datasets.
Visualice la corelacion entre las columnas de datos de entrada
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Cargar el dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear un mapa de calor para visualizar la correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de correlación entre características")
plt.show()
