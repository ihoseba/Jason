# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:59:13 2024

@author: joseangelperez
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Wine
wine_data = load_wine()
X, y = wine_data.data, wine_data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Representar gráficamente las fronteras de decisión
plt.figure(figsize=(12, 8))

# Gráfico SVM (kernel lineal)
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 12], c=y_test, cmap='viridis', edgecolor='k')
plt.title("SVM (kernel lineal)")
plt.xlabel("Alcohol")
plt.ylabel("Proline")
plt.legend(["Clase 0", "Clase 1", "Clase 2"])

# Gráfico SVM (kernel RBF)
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 12], c=y_test, cmap='viridis', edgecolor='k')
plt.title("SVM (kernel RBF)")
plt.xlabel("Alcohol")
plt.ylabel("Proline")
plt.legend(["Clase 0", "Clase 1", "Clase 2"])

plt.tight_layout()
plt.show()
