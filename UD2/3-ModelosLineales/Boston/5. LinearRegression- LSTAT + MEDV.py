# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:03:34 2024

@author: MarkelP

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


# Realizar regresión con RM + MEDV


















