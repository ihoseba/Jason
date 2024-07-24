# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:29:33 2024

@author: joseangelperez
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

# Cargar el dataset Wine
wine_data = load_wine()
dato1=print(wine_data.feature_names.index('alcohol'))
dato2=print(wine_data.feature_names.index('hue'))
X = wine_data.data[:, [dato1, dato2]]  # Seleccionar las columnas
y = wine_data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    "Árbol de decisión": DecisionTreeClassifier(max_depth=3)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy:.2f}")

    # Graficar las fronteras de decisión
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_test_scaled, y_test, clf=model, legend=2)
    plt.xlabel("flavanoids")
    plt.ylabel("total_phenols")
    plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
    plt.show()
