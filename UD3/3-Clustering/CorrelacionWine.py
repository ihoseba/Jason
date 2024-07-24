# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:28:03 2024

@author: joseangelperez

Copilot, genera un codigo python que partiendo de un csv llamado winequality-red.csv,
haga una matriz de correlacion de los diferentes parametros para poniendo leyenda 
y etiquetas en los parametros

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("winequality-red.csv")

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear un mapa de calor para visualizar la correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

# Agregar leyendas y etiquetas
plt.title("Matriz de Correlación para Parámetros de Vino Tinto")
plt.xlabel("Parámetros")
plt.ylabel("Parámetros")

# Mostrar el gráfico
plt.show()

