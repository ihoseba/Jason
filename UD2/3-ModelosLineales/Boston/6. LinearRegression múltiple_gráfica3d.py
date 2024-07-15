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

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

df = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(df, columns=column_names)

X = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns=['LSTAT', 'RM'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

lr = LinearRegression()
lr.fit(X_train, y_train)
res_pred = lr.predict(X_test)


print("W0", lr.intercept_)
print("W1", lr.coef_[0])
print("W2", lr.coef_[1])

# Fijamos el área de trazado
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

# Trazamos gráfica en 3D
ax.scatter(df['LSTAT'], df['RM'], df['MEDV'], c='b')

ax.set_xlabel("LSTAT", fontsize=15)

ax.set_ylabel("RM", fontsize=15)

ax.set_zlabel("MEDV", fontsize=15)

#  Ajustamos el espacio de representación y mostramos el hiperplano generado con el modelo
lstat_sup = np.arange(0, 40, 1)
rm_sup = np.arange(0, 10, 1)

lstat_sup, rm_sup = np.meshgrid(lstat_sup, rm_sup)

def z_medv(x, y):
    # W0 + X1 * W1 + X2 * W2
    coef_1 = lr.coef_[0]
    return (lr.intercept_ + coef_1 * x) + lr.coef_[1] * y

#z_medv = lambda x, y: (lr.intercept_ + lr.coef_[0] * x) + lr.coef_[1] * y

tmp = z_medv(lstat_sup, rm_sup)

ax.plot_surface(
    lstat_sup,
    rm_sup,
    tmp,
    rstride=1,
    cstride=1,
    color='red',
    alpha=0.4
)
