# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:26:16 2024

@author: joseangelperez
Copilot haz codigo python para visualizar el diagrama de correlacion de datos
de entrada de un fichero csv leido con pandas

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path_train = 'train.csv'

# Leer el archivo CSV (reemplaza 'tu_archivo.csv' con la ruta de tu archivo)
df = pd.read_csv(file_path_train)

data=df.iloc[:,[2,5,6,7,9]]

# Calcular la matriz de correlación
correlation_matrix = data.corr()

# Crear un mapa de calor para visualizar la correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Diagrama de Correlación")
plt.show()

