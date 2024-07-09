# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:51:42 2024

@author: joseangelperez
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_data = iris['data']

fig, ax = plt.subplots()
ax.plot(iris_data)