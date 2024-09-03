# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:29:33 2024

@author: joseangelperez

Copilot: en python para dataset digits: Encuentra el modelo que ofrezca el
mejor rendimiento al predecir utilizando el conjunto de datos "digits" de
sklearn.datasets
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
entre los siguientes:
    • SVC (con distintos kernels)
    • DecisionTreeClassifier
    • RandomForestClassifier
    • LogisticRegression
    • KNN
Plotea la matriz de confision de cada modelo tras hacr split de test y train
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Cargar el conjunto de datos "digits"
digits = load_digits()
X, y = digits.data, digits.target

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
