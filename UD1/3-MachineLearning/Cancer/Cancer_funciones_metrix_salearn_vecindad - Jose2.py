# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:21:23 2024

@author: joseangelperez
"""
# importar módulos
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# Cargamos los datos y dividimos en conjunto de entrenamiento y prueba

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    stratify=cancer.target,
    random_state=66
)

# Clasificación con K-nearest neighbors (k=3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Realizamos predicción

prediction_test = knn.predict(X_test)

# Obtenemos matriz de confusión

mat_conf = confusion_matrix(y_true=y_test, y_pred=prediction_test)

# Trazado de la matriz de confusión
fig, ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(mat_conf, cmap=plt.cm.Blues, alpha=.3)
for i in range(mat_conf.shape[0]):
    for j in range(mat_conf.shape[1]):
        val = mat_conf[i, j]
        ax.text(x=j, y=i, s=val, va='center', ha='center')
        plt.xlabel("Valores predichos")
        plt.ylabel("Valores reales")


# Exactitud de la clasificación
print("Exactitud:", round(accuracy_score(y_true=y_test, y_pred=prediction_test), 2))

plt.show()

# Buscar Optimo

num = []
exactitud = []
precision = []
recall = []
f1 = []

for i in range (1,10):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    # Realizamos predicción

    prediction_test = knn.predict(X_test)

    # Exactitud de la clasificación
    exact=round(accuracy_score(y_true=y_test, y_pred=prediction_test), 2)
    print("Exactitud:", exact)

    # Precision de la clasificación
    prec=round(precision_score(y_true=y_test, y_pred=prediction_test), 2)
    print("Precision:", exact)
    

    # Recall de la clasificación
    rec=round(recall_score(y_true=y_test, y_pred=prediction_test), 2)
    print("Recall:", exact)
    

    # F1 de la clasificación
    f=round(f1_score(y_true=y_test, y_pred=prediction_test), 2)
    print("F1:", exact)
    
    num.append(i)
    exactitud.append(exact)
    precision.append(prec)
    recall.append(rec)
    f1.append(f)

print(num,exactitud,precision,recall,f1)

fig, ax = plt.subplots()
ax.plot(num, exactitud, c='red')
ax.plot(num, precision, c='blue')
ax.plot(num, recall, c='green')
ax.plot(num, f1, c='orange')
# ax.xlabel("Exactitud")
# ax.ylabel("Exactitud")
# ax.title("Exactitud")
# ax.show()
