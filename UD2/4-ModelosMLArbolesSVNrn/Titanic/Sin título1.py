# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:58:17 2024

@author: joseangelperez

Copilot. Utiliza el dataset titanic disponible en kaggle. Clasifica el dataset utilizando dos columnas de datos mas correladas con el objetivo "survival":
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
La columna objetivo será: “survival”
Representa gráficamente las fronteras de decisión para cada método y Determina la exactitud alcanzada en cada caso. 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Titanic
titanic_data = fetch_openml(name="titanic", version=1, as_frame=True)
X, y = titanic_data.data, titanic_data.target

# Seleccionar las columnas más correlacionadas con "survival"
selected_columns = ["pclass", "fare"]
if all(col in X.columns for col in selected_columns):
    X_subset = X[selected_columns]
else:
    print("Las columnas seleccionadas no están presentes en el conjunto de datos.")

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)

# 1. SVM con kernel lineal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
print(f"Exactitud SVM (kernel lineal): {accuracy_svm_linear:.2f}")

# 2. SVM con kernel RBF
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
print(f"Exactitud SVM (kernel RBF): {accuracy_svm_rbf:.2f}")

# 3. Clasificador K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Exactitud KNN: {accuracy_knn:.2f}")

# 4. Árbol de decisión
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Exactitud Árbol de decisión: {accuracy_tree:.2f}")

# Representar gráficamente las fronteras de decisión (por ejemplo, SVM con kernel lineal)
plt.figure(figsize=(10, 6))
plt.scatter(X_test["Pclass"], X_test["Fare"], c=y_test, cmap='viridis', edgecolor='k')
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.title("Clasificación de pasajeros del Titanic")
plt.show()
