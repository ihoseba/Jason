# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:29:56 2024

@author: joseangelperez

Actividad práctica: algoritmo PCA
Seguramente recordarás el dataset Housing, sobre el que trabajamos cuando analizamos
la problemática de la regresión en Machine Learning. Este conjunto, disponible
en el método datasets de Scikit-learn, cuenta con 506 instancias y 14 características 
(recuerda que se añadía la mediana de los valores de las viviendas). Puedes 
cargarlo mediante la invocación a load_boston(). 
Para esta actividad práctica te planteamos que obtengas dicho dataset, que 
muestres información sobre cada dimensión y que analices sus varianzas. ¿Interesará
 escalar los datos? 
Aplica el algoritmo PCA y representa la varianza explicada en función del número
 de dimensiones. ¿Cuántas dimensiones requerirás para salvaguardar una varianza 
 en torno al 95 %? 
Obtén los valores de las proyecciones de los datos sobre los nuevos ejes y muestra
 las 5 primeras filas de esta matriz.


Copilot, en python 
carga el dataset Housing mediante la invocación a load_boston(). 
convierte los datos MEDV en 4 categorias
Escala todos los datos
muestra información sobre cada dimensión y analiza sus varianzas.
Aplica el algoritmo PCA y representa la varianza explicada 
en función del número de dimensiones.
Aplica en algoritmo PCS para tres dimensiones y representa graficamente
Obtén los valores de las proyecciones de los datos sobre los nuevos ejes y muestra
 las 5 primeras filas de esta matriz.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Cargar el dataset desde OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Convertir los datos MEDV en 4 categorías
df['MEDV'] = pd.qcut(df['MEDV'], 5, labels=False)

# Escalar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('MEDV', axis=1))

"""
# Mostrar información sobre cada dimensión y analizar sus varianzas
print(df.describe())
variances = df.var()
print("\nVarianzas:\n", variances)
"""

# Aplicar PCA
pca = PCA()
pca.fit(scaled_data)

# Representar la varianza explicada en función del número de dimensiones
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Número de dimensiones')
plt.ylabel('Varianza explicada acumulada')
plt.title('Varianza explicada acumulada por PCA')
plt.grid()
plt.show()

# Aplicar PCA para tres dimensiones y representar gráficamente
pca_3d = PCA(n_components=3)
pca_3d_data = pca_3d.fit_transform(scaled_data)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=df['MEDV'], cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="MEDV")
ax.add_artist(legend1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Proyección PCA en 3D')
plt.show()

# Obtener los valores de las proyecciones de los datos sobre los nuevos ejes y mostrar las 5 primeras filas
print("\nProyecciones de los datos sobre los nuevos ejes (primeras 5 filas):\n", pca_3d_data[:5])
