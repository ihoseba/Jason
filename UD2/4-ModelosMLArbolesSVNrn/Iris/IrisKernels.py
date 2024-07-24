# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:09:12 2024

@author: joseangelperez
"""

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Tomar las primeras dos características
y = iris.target
target_names = iris.target_names  # Nombres de las especies

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear modelos SVM con diferentes kernels
models1 = [
    svm.SVC(kernel="linear", C=1.0),
    svm.SVC(kernel="rbf", gamma=0.7, C=1.0),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0)
]
models = [
    svm.SVC(kernel="linear", C=2.0),
    svm.SVC(kernel="rbf", gamma=1.5, C=2.0),
    svm.SVC(kernel="poly", degree=6, gamma="auto", C=2.0)
]

# Entrenar los modelos
for clf in models:
    clf.fit(X_train, y_train)

# Calcular la precisión en el conjunto de prueba
accuracies = [accuracy_score(y_test, clf.predict(X_test)) for clf in models]

# Mostrar gráficamente las regiones de decisión
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["SVC con kernel lineal", "SVC con kernel RBF", "SVC con kernel polinomial"]

for clf, title, ax, acc in zip(models, titles, axes, accuracies):
    plot_decision_regions(X_train, y_train, clf, ax=ax, legend=2)
    ax.set_title(f"{title}\nPrecisión: {acc:.2f}")
    ax.set_xlabel("Longitud del sépalo")
    ax.set_ylabel("Ancho del sépalo")
    ax.legend(title="Especies", labels=target_names)

plt.show()

