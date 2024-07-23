# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:15:31 2024

@author: joseangelperez
Actividad práctica: clasificación de elementos mediante perceptrón

Siguiendo el esquema de programación que acabas de estudiar, trata de clasificar los elementos
del dataset iris mediante un perceptrón. Considera, de nuevo, la longitud de los sépalos y la de
los pétalos. Compara esta solución con la que obtuviste al aplicar el algoritmo SVM. ¿Qué
diferencia aprecias respecto a la clasificación realizada en el dataset forge()?
Responde a la cuestión planteada y:
    
"""

import seaborn as sns
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Carga el dataset Iris
iris = sns.load_dataset("iris")

iris["label"] = iris.species.astype("category").cat.codes
# X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
X = iris[["sepal_length", "petal_length"]]
y = iris["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

sc=StandardScaler()
sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn = Perceptron(
    max_iter=40,
    eta0=1,
    random_state=1
)
ppn.fit(X_train_std,y_train)

predicciones=ppn.predict(X_test_std)

print("Entradas mal clasificadas:%d"%(y_test !=predicciones).sum())

accuracy = accuracy_score(y_true = y_test, y_pred = predicciones, normalize = True)
accuracy=round(accuracy,2)

print(f"La exactitud del test es: {100*accuracy}%")

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined_std, y_combined, ppn, X_highlight=X_test_std,
                      legend=2)

# Agrega etiquetas y título
plt.xlabel('Longitud de sépalos [cm]')
plt.ylabel('Longitud de pétalos [cm]')
plt.title('Matriz de decisión del Perceptron en Iris')
plt.show()