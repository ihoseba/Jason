# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:20:25 2024

@author: MarkelP
"""

# importar módulos
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# Cargamos los datos y dividimos en conjunto de entrenamiento y prueba

cancer = load_breast_cancer()
print(type(cancer))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data,
    cancer.target,
    stratify=cancer.target,
    random_state=66
)


# Clasificación con K-nearest neighbors (k=3)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Realizamos predicción

prediction_res = knn.predict_proba(X_test)

prediction_test = prediction_res[:,1]

# Cogemos el valor AUC

auc = roc_auc_score(y_true=y_test, y_score=prediction_test)

print('AUC - Conjunto de entrenamiento', auc * 100)

# Llamamos a la  función que genera la curva de ROC
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=prediction_test)

# Dibujamos la línea AUC=.5
plt.plot([0, 1], [0, 1], linestyle='--')

# Dibujamos la curva de ROC
plt.plot(fpr, tpr, marker='.')
plt.xlabel('Tasa de falsos positivos / FPR')
plt.ylabel('Sensibilidad / TPR')

plt.show()













