# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:17:57 2024

@author: joseangelperez

Copilot genera un codigo en python que utiliza el dataset Wine disponible en sklearn.datasets.
Visualice en varios formatos la corelacion entre las columnas de datos de entrada y de salida

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Cargar el dataset Wine
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target  # Agregar la columna de salida (target)

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Visualizar la correlación en un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de correlación entre características")
plt.show()
