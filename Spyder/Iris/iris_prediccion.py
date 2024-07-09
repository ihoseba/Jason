# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:45:07 2024

@author: joseangelperez
"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas import plotting
from matplotlib import pyplot as plt
""" import seaborn as sns """

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


plt.style.use('ggplot')

iris_conj = load_iris(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    iris_conj['data'],
    iris_conj['target'],
    random_state=0,  # Para que la semilla siempre sea la misma 
    # y los resultados sean comparables
    # test_size=25,
    train_size=75
)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

dcorr = plotting.scatter_matrix(
    X_train,
    c=y_train,
    figsize=(15, 15),
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8
)

knn = KNeighborsClassifier(n_neighbors=1)
# Entrenar modelo
knn.fit(X_train, y_train)


#realizamos prediccion de pruebas

prediccion_test = knn.predict(X_test)

# comparar resultados
print('la exactitud del modelo es')
exactitud = np.mean(prediccion_test==y_test)
exactitud_knn=knn.score(X_test, y_test)

print(exactitud_knn)


