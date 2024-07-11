# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:10:32 2024

@author: joseangelperez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

df = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(df, columns=column_names)

#
# Revisa las correlaciones entre variables e identifica qué otras dos 
# variables utilizarías, además, para tratar de predecir el valor de
# las casas del dataset Housing. 
#

"""
data_full = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(data_full, columns=column_names)

columnas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

vals = df[columnas].values
t_vals = vals.T


matriz_corr = np.corrcoef(t_vals)

mapa_calor = sns.heatmap(matriz_corr,
                         cbar=True,
                         annot=True,
                         fmt='.2f',
                         square=True,
                         annot_kws={'size': 14},
                         yticklabels=columnas,
                         xticklabels=columnas)
"""

#
# Posteriormente, genera el nuevo modelo y calcula cuáles son los nuevos 
# coeficientes de determinación y su nuevo error cuadrático medio. 
#

X = pd.DataFrame(np.c_[df['LSTAT'], df['RM'], df['INDUS'], df['PTRATIO']],
                 columns=['LSTAT', 'RM', 'INDUS', 'PTRATIO'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


print("W0", lr.intercept_)
print("W1 LSTAT", lr.coef_[0])
print("W2 RM", lr.coef_[1])
print("W3 INDUS", lr.coef_[2])
print("W4 PTRATIO", lr.coef_[3])

# Valor del error cuadrático medio
print("Error cuadrático medio:", mean_squared_error(
    y_true=y_test,
    y_pred=y_pred
))

#
# Finalmente, representa el diagrama predicción de precios vs residuos
# e interpreta el resultado
#

y_test_pred = lr.predict(X_test)

"""
plt.xlabel("precios reales")
plt.ylabel("prediccion de precios")
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([0,50], [0,50], color='red', linewidth=2)
"""

#Obtenemos la salida gráfica solicitada
plt.xlabel("precios previstos")
plt.ylabel("diferencia de precios")
"""
plt.scatter(y_test_pred, y_test,c='steelblue',marker='o',
            edgecolor='white', label='Datos de entrenamiento')
"""
plt.scatter(y_test_pred, y_test_pred-y_test,c='limegreen',marker='s',
            edgecolor='white',label='Datos de prueba')

plt.show()


