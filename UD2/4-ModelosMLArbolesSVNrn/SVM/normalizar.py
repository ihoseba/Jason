# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:43:34 2024

@author: joseangelperez
"""

import pandas as pd
import numpy as np

# Gráficas
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Preparación de los datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Modelado
from sklearn.svm import LinearSVC
import mglearn

# Medición de resultados
from sklearn.metrics import accuracy_score

# Accedemos a los datos y los cargamos en un dataframe
url = "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/ESL.mixture.csv"
datos = pd.read_csv(url)

# Creamos el dataframe con las coordeadas de puntos
X = pd.DataFrame(np.c_[datos['X1'], datos['X2']], columns=['X1', 'X2'])

# Cargamos la columna objetivo (clasificación real de los valores)
y = datos['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)


# Realizamos pre-escalado de características
sc = StandardScaler()
sc.fit(X_train)

# Aplicamos la transformación para que los datos mantengan la misma media
# y la misma desviación estandar
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Aplicamos el algoritmo SVM sobre los datos normalizados

clf = LinearSVC()
clf = SVC(kernel='rbff",random_state =1,
clf.fit(X_train_std, y_train)

# Creamos arrays bidimensionales con los datos del entrenamiento y test
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Representamos el resultado con plot_decision_regions
plot_decision_regions(X_combined_std, y_combined, clf, X_highlight=X_test_std)

# Analizamos la exactitud del modelos sobre los datos de prueba
predicciones = clf.predict(X_test_std)

accuracy = accuracy_score(y_true=y_test, y_pred=predicciones, normalize = True)

print("")
print("La exactitud del test es:", 100 * accuracy, "%")


