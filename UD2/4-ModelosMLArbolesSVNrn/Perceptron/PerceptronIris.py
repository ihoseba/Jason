# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:10:00 2024

@author: joseangelperez

Actividad práctica: clasificación de elementos mediante perceptrón

Siguiendo el esquema de programación que acabas de estudiar, trata de clasificar los elementos
del dataset iris mediante un perceptrón. Considera, de nuevo, la longitud de los sépalos y la de
los pétalos. Compara esta solución con la que obtuviste al aplicar el algoritmo SVM. ¿Qué
diferencia aprecias respecto a la clasificación realizada en el dataset forge()?
Responde a la cuestión planteada y:
    
Copilot, en python utilizando clasifica los elementos del dataset iris 
mediante un perceptrón.
Considera, la longitud de los sépalos y la de los pétalos.
muestra la plot_decission_matrix, incluyendo todas las leyendas


"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# Carga el dataset Iris
iris = sns.load_dataset("iris")

# Transforma la columna "species" en una etiqueta numérica
iris["label"] = iris.species.astype("category").cat.codes

# Entrenamiento del perceptrón
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["label"]
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X, y)

# Visualización de la matriz de decisión
plot_decision_regions(X.values, y.values, clf=perceptron, legend=2)
plt.xlabel("Longitud de sépalos")
plt.ylabel("Longitud de pétalos")
plt.title("Matriz de decisión del perceptrón")
plt.show()
