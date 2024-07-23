# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:53:55 2024

@author: joseangelperez

Copilot. Utiliza el dataset titanic disponible en kaggle.
Clasifica el dataset utilizando dos columnas de datos fare y pclass
con el objetivo "survival":
    • SVM con kernel lineal, RBF y polyinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
La columna objetivo será: “survival”
Representa gráficamente las fronteras de decisión para cada método y
imprime la precision alcanzada en cada caso. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carga el conjunto de datos Titanic (reemplaza con tu ruta local o descarga desde Kaggle)
df = pd.read_csv('train.csv')

# Elimina filas con valores nulos
df.dropna(subset=['Fare', 'Pclass', 'Survived'], inplace=True)

# Codifica la columna 'pclass' como variables numéricas
le = LabelEncoder()
df['Pclass'] = le.fit_transform(df['Pclass'])

# Supongamos que ya tienes el DataFrame cargado con las columnas 'fare', 'pclass' y 'survival'
# Aquí realizamos el preprocesamiento (eliminación de nulos, codificación de 'pclass', etc.)

# Divide los datos en conjuntos de entrenamiento y prueba
X = df[['Fare', 'Pclass']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM con kernel lineal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)

# SVM con kernel RBF
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)

# SVM con kernel polinomial
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)
y_pred_svm_poly = svm_poly.predict(X_test)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Árbol de decisión
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Calcula la precisión
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

print(f"Precisión SVM (lineal): {accuracy_svm_linear:.2f}")
print(f"Precisión SVM (RBF): {accuracy_svm_rbf:.2f}")
print(f"Precisión SVM (polinomial): {accuracy_svm_poly:.2f}")
print(f"Precisión KNN: {accuracy_knn:.2f}")
print(f"Precisión Árbol de decisión: {accuracy_tree:.2f}")

# Representa gráficamente las fronteras de decisión (solo para SVM lineal)
plt.figure(figsize=(8, 6))
plt.scatter(X['Fare'], X['Pclass'], c=y, cmap='coolwarm', edgecolor='k', s=50)
plt.xlabel('Fare')
plt.ylabel('Pclass')
plt.title('Fronteras de decisión (SVM lineal)')
plt.show()

