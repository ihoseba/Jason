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

# Fijamos el área de trazado
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

# Trazamos gráfica en 3D
ax.scatter(df['LSTAT'], df['RM'], df['MEDV'], c='b')

ax.set_xlabel("LSTAT", fontsize=15)

ax.set_ylabel("RM", fontsize=15)

ax.set_zlabel("MEDV", fontsize=15)
