# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:24:50 2024

@author: MarkelP
"""
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters

datos, filas, columnas = make_biclusters(shape=(100, 100), n_clusters=3, noise=5, shuffle=False, random_state=0)
plt.matshow(datos)