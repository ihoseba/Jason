# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas import plotting
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Cambiamos tema visual matplotlib
plt.style.use('ggplot')

# Cargamos iris como dataframe
iris_conj = load_iris(as_frame=True)

# Dividimos el conjunto de datos en entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(
    iris_conj['data'],
    iris_conj['target'],
    random_state=1,
    #test_size=.5, # Tamaño del set de entrenamiento
    #train_size=.5 # Tamaño del set de testing
)

'''
dcorr = plotting.scatter_matrix(
    X_train, # Datos a visualizar
    c=y_train, # Etiquetado de datos
    figsize=(15, 15), # Dimensiones de la ventan
    hist_kwds={'bins': 20}, # Numero de rangos numéricos de los histogramas
    s=100, # Tamaño puntos de dispersión
    alpha=0.8 # Transparencia a aplicar
)
'''

# Instanciamos el modelo que vamos a utililzar
# Params interesantes:
    # n_neighbors = número de vecinos
    # weights = pesos; "uniform" por defecto
knn = KNeighborsClassifier(n_neighbors=10)
 


# Entrenar modelo
knn.fit(X_train, y_train)

# Predecimos las etiquetas de X_test (dataset de testing)
prediccion_test = knn.predict(X_test)

# Generamos la matriz de confusión: comparando test vs predicción
mat_conf = confusion_matrix(y_true=y_test, y_pred=prediccion_test)

etiquetas = iris_conj['target_names']
# Trazado de la matriz de confusión
fig, ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(mat_conf, cmap=plt.cm.Blues, alpha=.3)
print("mat_conf.shape", mat_conf.shape)
for i in range(mat_conf.shape[0]):
    for j in range(mat_conf.shape[1]):
        ax.text(x=j, y=i, s=mat_conf[i, j], va='center', ha='center')
        plt.xlabel("Valores predichos")
        plt.ylabel("Valores reales")

ax.set_xticks(np.arange(len(etiquetas)))
ax.set_yticks(np.arange(len(etiquetas)))
ax.set_xticklabels(etiquetas)
ax.set_yticklabels(etiquetas)

fig.show()