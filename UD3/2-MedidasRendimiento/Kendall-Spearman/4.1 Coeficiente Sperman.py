# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:29:40 2024

@author: MarkelP
"""

from scipy.stats import spearmanr

# Datos de trabajo

x = [1, 2, 3, 5, 7, 8, 9,  10]
y = [1, 3, 4, 2, 8, 7, 10, 9]

coef_sp, p = spearmanr(x, y)

print('El valor del coefiniciente de Sperman', coef_sp)
print('El valor de p, es:', p)

alpha = 0.05
if p > alpha:
    print("Las dos listas no están correcilacionadas")

else:
    print('las dos listas están correlacionadas')
