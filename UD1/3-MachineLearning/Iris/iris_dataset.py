# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:15:27 2024

@author: joseangelperez
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()

iris_data=iris['data']

sub_dat=iris_data[:10]

columnas = iris['feature_names']

target_col=iris['target']

target_col_names=iris['target_names']

frame=iris['frame']

fix, ax = plt.subplots()
ax.plot(iris_data)

plt.show()
