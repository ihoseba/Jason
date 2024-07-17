# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:15:05 2024

@author: joseangelperez


Utiliza el dataset Wine disponible en sklearn.datasets. Clasifica el dataset utilizando:
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
Determina la exactitud alcanzada en cada caso y comenta los resultados. Representa gráficamente las fronteras de decisión para cada método)
    
Copilot genera un codigo en python que utiliza el dataset Wine disponible en sklearn.datasets.
selecciona flavanoids y diluted_wines como datos de entrada y entrena modelos
SVM con kernel lineal, RBF y polinomial.
Un clasificador K-Nearest Neighbors (KNN).
Un árbol de decisión.
Representa gráficamente las fronteras de decisión para cada método y pone en la leyenda la exactitud alcanzada en cada caso

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Cargar el dataset Wine
wine_data = load_wine()
X = wine_data.data[:, [6, 12]]  # Seleccionar las columnas "flavanoids" y "diluted_wines"
y = wine_data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos
models1 = {
    "SVM (Lineal)": SVC(kernel="linear", C=10),
    "SVM (RBF)": SVC(kernel="rbf", gamma=6, C=8.0),
    "SVM (Polinomial)": SVC(kernel="poly", degree=6, gamma=4, C=2),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=3)
}
models = {
    "SVM (Lineal)": SVC(kernel="linear"),
    "SVM (RBF)": SVC(kernel="rbf"),
    "SVM (Polinomial)": SVC(kernel="poly",degree=3),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=3)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.2f}")

    # Graficar las fronteras de decisión
    plt.figure(figsize=(8, 6))
    cmap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    xx, yy = np.meshgrid(np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 100),
                         np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=cmap, edgecolor="k")
    plt.xlabel("flavanoids")
    plt.ylabel("diluted_wines")
    plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
    plt.show()
