# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:22:59 2024

@author: joseangelperez
Wine recognition dataset
------------------------

**Data Set Characteristics:**

:Number of Instances: 178
:Number of Attributes: 13 numeric, predictive attributes and the class
:Attribute Information:
    - Alcohol
    - Malic acid
    - Ash
    - Alcalinity of ash
    - Magnesium
    - Total phenols
    - Flavanoids
    - Nonflavanoid phenols
    - Proanthocyanins
    - Color intensity
    - Hue
    - OD280/OD315 of diluted wines
    - Proline
    - class:
        - class_0
        - class_1
        - class_2

:Summary Statistics:

============================= ==== ===== ======= =====
                                Min   Max   Mean     SD
============================= ==== ===== ======= =====
Alcohol:                      11.0  14.8    13.0   0.8
Malic Acid:                   0.74  5.80    2.34  1.12
Ash:                          1.36  3.23    2.36  0.27
Alcalinity of Ash:            10.6  30.0    19.5   3.3
Magnesium:                    70.0 162.0    99.7  14.3
Total Phenols:                0.98  3.88    2.29  0.63
Flavanoids:                   0.34  5.08    2.03  1.00
Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
Proanthocyanins:              0.41  3.58    1.59  0.57
Colour Intensity:              1.3  13.0     5.1   2.3
Hue:                          0.48  1.71    0.96  0.23
OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
Proline:                       278  1680     746   315
============================= ==== ===== ======= =====

:Missing Attribute Values: None
:Class Distribution: class_0 (59), class_1 (71), class_2 (48)
:Creator: R.A. Fisher
:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
:Date: July, 1988

This is a copy of UCI ML Wine recognition datasets.
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

The data is the results of a chemical analysis of wines grown in the same
region in Italy by three different cultivators. There are thirteen different
measurements taken for different constituents found in the three types of
wine.

Original Owners:

Forina, M. et al, PARVUS -
An Extendible Package for Data Exploration, Classification and Correlation.
Institute of Pharmaceutical and Food Analysis and Technologies,
Via Brigata Salerno, 16147 Genoa, Italy.

Citation:

Lichman, M. (2013). UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
School of Information and Computer Science.

.. dropdown:: References

    (1) S. Aeberhard, D. Coomans and O. de Vel,
    Comparison of Classifiers in High Dimensional Settings,
    Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of
    Mathematics and Statistics, James Cook University of North Queensland.
    (Also submitted to Technometrics).

    The data was used with many others for comparing various
    classifiers. The classes are separable, though only RDA
    has achieved 100% correct classification.
    (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data))
    (All results using the leave-one-out technique)

    (2) S. Aeberhard, D. Coomans and O. de Vel,
    "THE CLASSIFICATION PERFORMANCE OF RDA"
    Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of
    Mathematics and Statistics, James Cook University of North Queensland.
    (Also submitted to Journal of Chemometrics).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions


# Utiliza el dataset Wine disponible en sklearn.datasets.
    
# Cargar el conjunto de datos Wine
wine_data = load_wine()
X, y = wine_data.data, wine_data.target

# Normalizar las características
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

# Dividir los datos normalizados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)


# Clasifica el dataset utilizando:
#    • SVM con kernel lineal, RBF y polinomial.
# Crear y entrenar los clasificadores
svm_linear = SVC(kernel='linear', C=200)
svm_linear.fit(X_train, y_train)
# Determina la exactitud alcanzada en cada caso
# Calcular la precisión en los conjuntos de prueba
accuracy_svm_linear = accuracy_score(y_test, svm_linear.predict(X_test))
# Crear y entrenar los clasificadores
svm_rbf = SVC(kernel='rbf', gamma=6, C=8.0)
svm_rbf.fit(X_train, y_train)
# Determina la exactitud alcanzada en cada caso
# Calcular la precisión en los conjuntos de prueba
accuracy_svm_rbf = accuracy_score(y_test, svm_rbf.predict(X_test))
# Crear y entrenar los clasificadores
svm_poly = SVC(kernel='poly', degree=6, gamma=4, C=2)
svm_poly.fit(X_train, y_train)
# Determina la exactitud alcanzada en cada caso
# Calcular la precisión en los conjuntos de prueba
accuracy_svm_poly = accuracy_score(y_test, svm_poly.predict(X_test))
# Clasifica el dataset utilizando:
#    • Un clasificador K-Nearest Neighbors (KNN).
# Crear y entrenar los clasificadores
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Determina la exactitud alcanzada en cada caso
# Calcular la precisión en los conjuntos de prueba
accuracy_knn = accuracy_score(y_test, knn.predict(X_test))
# Clasifica el dataset utilizando:
#    • Un árbol de decisión.
# Crear y entrenar los clasificadores
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
# Determina la exactitud alcanzada en cada caso
# Calcular la precisión en los conjuntos de prueba
accuracy_tree = accuracy_score(y_test, decision_tree.predict(X_test))

# Representa gráficamente las fronteras de decisión para cada método)

# Graficar las fronteras de decisión
plt.figure(figsize=(12, 8))

# SVM con kernel lineal
plt_lineal=plt.subplot(3, 2, 1)
plt.title(f"SVM (Lineal) - Precisión: {accuracy_svm_linear:.2f}")
# plot_decision_regions(X_train, y_train, svm_linear, plt_lineal, legend=2)
# Representa las fronteras de decisión aquí

# SVM con kernel RBF
plt.subplot(3, 2, 2)
plt.title(f"SVM (RBF) - Precisión: {accuracy_svm_rbf:.2f}")
# Representa las fronteras de decisión aquí

# SVM con kernel RBF
plt.subplot(3, 2, 3)
plt.title(f"SVM (poly) - Precisión: {accuracy_svm_poly:.2f}")
# Representa las fronteras de decisión aquí

# KNN
plt.subplot(3, 2, 4)
plt.title(f"KNN - Precisión: {accuracy_knn:.2f}")
# Representa las fronteras de decisión aquí

# Árbol de decisión
plt.subplot(3, 2, 5)
plt.title(f"Árbol de decisión - Precisión: {accuracy_tree:.2f}")
# Representa las fronteras de decisión aquí

plt.tight_layout()
plt.show()

"""
    plot_decision_regions(X_train, y_train, clf, ax=ax, legend=2)
    ax.set_title(f"{title}\nPrecisión: {acc:.2f}")
    ax.set_xlabel("Longitud del sépalo")
    ax.set_ylabel("Ancho del sépalo")
    ax.legend(title="Especies", labels=target_names)

"""





# y comenta los resultados.


