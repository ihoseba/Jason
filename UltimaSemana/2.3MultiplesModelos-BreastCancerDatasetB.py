# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:19:44 2024

@author: joseangelperez

Encuentra el modelo que ofrezca el mejor rendimiento al predecir utilizando 
el conjunto de datos "breast cancer" de sklearn.datasets (load_breast_cancer
— scikit-learn 1.5.1 documentation) entre los siguientes:
    • SVC (con distintos kernels)
    • DecisionTreeClassifier
    • RandomForestClassifier
    • LogisticRegression
    • KNN

Adjunta capturas que muestren los resultados de precisión de cada uno de los
modelos y justifica tu elección.

Copilot, genera codigo en python para cargar en dataset "breast cancer" de
sklearn.datasets (load_breast_cancer — scikit-learn 1.5.1 documentation) ,
posteriormente evaluar lo siguientes modelos de prediccion:
    • SVC (con distintos kernels)
    • DecisionTreeClassifier
    • RandomForestClassifier
    • LogisticRegression
    • KNN
Muestra los resultados previstos dividiendo en train y test para cada modelo.
Tambien lista la precision de cada modelo indicando cual es el mas preciso
Plotea la matriz de confision de cada modelo tras hacr split de test y train

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Cargar el conjunto de datos "breast cancer"
data = load_breast_cancer()
X, y = data.data, data.target

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

# Evaluar el rendimiento de cada modelo y plotear la matriz de confusión
resultados = {}
plt.figure(figsize=(20, 12))

for i, (nombre, modelo) in enumerate(modelos.items(), 1):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    resultados[nombre] = score
    
    # Plotear la matriz de confusión
    plt.subplot(3, 3, i)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{nombre} (Accuracy: {score:.4f})")
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()

# Mostrar los resultados
for nombre, score in resultados.items():
    print(f"{nombre}: {score:.4f}")

# Encontrar el mejor modelo
mejor_modelo = max(resultados, key=resultados.get)
print(f"\nEl mejor modelo es: {mejor_modelo} con una precisión de {resultados[mejor_modelo]:.4f}")