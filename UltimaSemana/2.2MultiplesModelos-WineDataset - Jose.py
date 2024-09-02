# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:11:59 2024

@author: joseangelperez

Encuentra el modelo que ofrezca el mejor rendimiento al predecir utilizando
el conjunto de datos "wine" de sklearn.datasets (load_wine — scikit-learn
1.5.1 documentation) entre los siguientes:
    • SVC (con distintos kernels)
    • DecisionTreeClassifier
    • RandomForestClassifier
    • LogisticRegression
    • KNN

Adjunta capturas que muestren los resultados de precisión de cada uno de
los modelos y justifica tu elección.

Copilot genera codigo python para elvaluar el rendimiento de los siguientes
modelos utilizando el conjunto de datos "wine" de sklearn.datasets (load_wine
— scikit-learn 1.5.1 documentation) entre los siguientes:
    • SVC (con distintos kernels)
    • DecisionTreeClassifier
    • RandomForestClassifier
    • LogisticRegression
    • KNN
Encuentra el modelo que ofrezca el mejor rendimiento al predecir y plotealo en
una grafica

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Cargar el conjunto de datos "wine"
wine = load_wine()
X, y = wine.data, wine.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir los modelos
modelos = {
    "SVC (linear)": SVC(kernel='linear'),
    "SVC (rbf)": SVC(kernel='rbf'),
    "SVC (poly)": SVC(kernel='poly'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier()
}

# Evaluar el rendimiento de cada modelo utilizando cross_val_score
resultados = {}
for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=5)
    resultados[nombre] = np.mean(scores)

# Mostrar los resultados
for nombre, score in resultados.items():
    print(f"{nombre}: {score:.4f}")

# Encontrar el mejor modelo
mejor_modelo = max(resultados, key=resultados.get)
print(f"\nEl mejor modelo es: {mejor_modelo} con una precisión de {resultados[mejor_modelo]:.4f}")

# Ploteo de los resultados
plt.figure(figsize=(10, 6))
plt.bar(resultados.keys(), resultados.values(), color='skyblue')
plt.xlabel('Modelos')
plt.ylabel('Precisión')
plt.title('Rendimiento de Modelos en el Conjunto de Datos "Wine"')
plt.xticks(rotation=45)
plt.show()

