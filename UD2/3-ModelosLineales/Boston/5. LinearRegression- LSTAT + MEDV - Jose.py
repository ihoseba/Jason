# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:03:34 2024

@author: MarkelP

 The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
 prices and the demand for clean air', J. Environ. Economics & Management,
 vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
 ...', Wiley, 1980.   N.B. Various transformations are used in the table on
 pages 244-261 of the latter.

 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# ++++++++++++++++
# Realizar regresión con SSTAT + MEDV
# ++++++++++++++++

# Preparar el primer array de regresión
lstat = np.array([df['LSTAT']])
lstat = np.transpose(lstat)

# Preparamos el segundo array para la regresión
medv = np.array([df['MEDV']])
medv = np.transpose(medv)

X_train, X_test, y_train, y_test = train_test_split(lstat, medv, random_state=20)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_pred2 = lr.predict([[0]])

# Valores de los coeficientes
print("W1", lr.coef_)
print("W0", lr.intercept_)

# Salida gráfica
plt.scatter(X_test, y_test, color='black')

plt.xlabel("LSTAT")
plt.ylabel("MEDV")

plt.plot(X_test, y_pred, color='blue', linewidth=3)


# Valor del coeficiente de determinación del conjunto de entrenamiento
coef_training = round(lr.score(X_train, y_train), 2)

# Valor del coeficiente de determinación del conjunto de testing
coed_testing = round(lr.score(X_test, y_test), 2)

# ++++++++++++++++
# Realizar regresión con RM + MEDV
# ++++++++++++++++

# Preparar el primer array de regresión
rm = np.array([df['RM']])
rm = np.transpose(rm)

# Preparamos el segundo array para la regresión
medv = np.array([df['MEDV']])
medv = np.transpose(medv)

X_train, X_test, y_train, y_test = train_test_split(rm, medv, random_state=20)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_pred2 = lr.predict([[0]])

# Valores de los coeficientes
print("W1", lr.coef_)
print("W0", lr.intercept_)

# Salida gráfica
plt.scatter(X_test, y_test, color='red')

plt.xlabel("RM/LSTAT")
plt.ylabel("MEDV")

plt.plot(X_test, y_pred, color='blue', linewidth=3)


# Valor del coeficiente de determinación del conjunto de entrenamiento
coef_training = round(lr.score(X_train, y_train), 2)

# Valor del coeficiente de determinación del conjunto de testing
coed_testing = round(lr.score(X_test, y_test), 2)

















