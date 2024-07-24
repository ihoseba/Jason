# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:01:30 2024

@author: joseangelperez

Basado en WineAplicacionModelosC

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
mejor_a = 0
mejor_b = 0
mejor_s=0
for a in range(12):
    for b in range(12):
        X = wine_data.data[:, [a, b]]  # Seleccionar las columnas "flavanoids" y "total_phenols"
        y = wine_data.target
        dato1=wine_data.feature_names[a]
        dato2=wine_data.feature_names[b]
        print(dato1,dato2)
        
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
        
        Suma=0
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            Suma+=accuracy
            print(f"{name}: Accuracy = {accuracy:.2f}")
        """
            # Graficar las fronteras de decisión
            plt.figure(figsize=(8, 6))
            plot_decision_regions(X_test_scaled, y_test, clf=model, legend=2)
            plt.xlabel(dato1)
            plt.ylabel(dato2)
            plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
            plt.show()
        """
        print("Suma ",Suma)
        if Suma>mejor_s:
            mejor_s=Suma
            mejor_a=a
            mejor_b=b
print("Mejor",mejor_a,mejor_b,mejor_s)
