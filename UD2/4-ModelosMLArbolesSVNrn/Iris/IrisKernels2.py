# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:09:12 2024

@author: joseangelperez
"""
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data  # Tomar todas las características (4 columnas)
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=81)

# Crear modelos SVM con diferentes kernels
models = [
    svm.SVC(kernel="linear", C=C)
    for C in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 10.0]
]

# Entrenar los modelos y calcular la precisión en el conjunto de prueba
accuracies = []
for clf in models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Mostrar gráficamente la precisión frente al valor de C
plt.plot([C for C in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 10.0]], accuracies, marker="o")
plt.xlabel("Valor de C")
plt.ylabel("Precisión (Exactitud)")
plt.title("Precisión vs. Valor de C para diferentes kernels")
plt.grid(True)
plt.show()