# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:08:35 2024

@author: MarkelP
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score


num_clusters = (4, 3)
datos, filas, columnas = make_checkerboard(shape=(30, 50), n_clusters=num_clusters, noise=10, shuffle=False, random_state=0)
plt.matshow(datos, cmap=plt.cm.Greens)
plt.title("Conjunto original")
print(datos)
