# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:29:29 2024

@author: MarkelP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

df = np.hstack([data, target.reshape(-1, 1)])

casas = pd.DataFrame(df, columns=column_names)

#Preparación de los dataframe para la regresión
X = pd.DataFrame(np.c_[casas['LSTAT'], casas['RM'],casas['PTRATIO'],casas['INDUS']],
columns = ['LSTAT','RM','PTRATIO','INDUS'])
y=casas['MEDV']
#Realizamos el entrenamiento y lanzamos las predicciones
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)
lr=LinearRegression().fit(X_train,y_train)
y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)
#Valor de los coeficientes de determinación y del error cuadrático medio
print('Coeficiente de determinación del conjunto de entrenamiento: %.4f' % lr.score(X_train, y_train))
print('Coeficiente de determinación del conjunto de pruebas: %.4f' % lr.score(X_test, y_test))
print("Error cuadrático medio:",round(mean_squared_error(y_test, y_test_pred),4))
#Obtenemos la salida gráfica solicitada
plt.scatter(y_train_pred, y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',
label='Datos de entrenamiento')
plt.scatter(y_test_pred, y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',
label='Datos de prueba')