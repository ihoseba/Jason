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