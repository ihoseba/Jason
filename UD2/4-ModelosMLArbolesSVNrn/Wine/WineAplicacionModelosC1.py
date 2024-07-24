# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:34:10 2024

@author: joseangelperez

Utiliza el dataset Wine disponible en sklearn.datasets. Clasifica el dataset utilizando:
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
Determina la exactitud alcanzada en cada caso y comenta los resultados. Representa gráficamente las fronteras de decisión para cada método)
    
Copilot genera un codigo en python que utiliza el dataset Wine disponible en sklearn.datasets.
selecciona flavanoids y total_phenols como datos de entrada y entrena modelos
SVM con kernel lineal, RBF y polinomial.
Un clasificador K-Nearest Neighbors (KNN).
Un árbol de decisión.
Representa gráficamente las fronteras de decisión para cada método utilizando plot_decision_regions
y pon en la leyenda la exactitud alcanzada en cada caso
    
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import Perceptron

# Cargar el dataset Wine
wine_data = load_wine()
index_x=wine_data.feature_names.index("alcohol")
index_y=wine_data.feature_names.index("hue")
X = wine_data.data[:, [index_x, index_y]]  # Seleccionar las columnas "flavanoids" y "total_phenols"
y = wine_data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos
models = {
    "SVM (Lineal)": SVC(kernel="linear"),
    "SVM (RBF)": SVC(kernel="rbf"),
    "SVM (Polinomial)": SVC(kernel="poly", degree=3),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=3),
    "Perceptron": Perceptron(max_iter=40,eta0=1,random_state=1)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.2f}")

    # Graficar las fronteras de decisión
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_test_scaled, y_test, clf=model, legend=2)
    plt.xlabel(wine_data.feature_names[index_x])
    plt.ylabel(wine_data.feature_names[index_y])
    plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
    plt.show()
