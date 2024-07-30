# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:58:57 2024

@author: joseangelperez
"""

import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Generamos el dataset (100 elementos)
X, y = mglearn.datasets.make_wave(100)

# Creamos conjunto de entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

# Ajustamos el modelo según los planteamiento OLS (Ordinary Least Squares)
# El ajuste se realiza sobre los datos de entrenamiento
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Valores de los coeficientes
print("Coeficiente w1", lr.coef_)
print("Coeficiente w0", lr.intercept_)

# Valor del error cuadrático medio
print("Error cuadrático medio:", mean_squared_error(
    y_true=y_test,
    y_pred=y_pred
))

# Valores del coeficiente de determinación 
print("Valor del coeficiente de determinación del conjunto de train:",
      round(lr.score(X_train, y_train), 3)
)

# Valores del coeficiente de determinación 
print("Valor del coeficiente de determinación del conjunto de test:",
      round(lr.score(X_test, y_test), 3)
)

import numpy as np
from seaborn import scatterplot
plt.figure(figsize=(10, 7))
scatterplot(x=X_test[:, 0], y=X_test[:, 1],c=y_pred)
plt.title(f"Datos Reales con {len(np.unique(y_pred))} clusters")
plt.show()
