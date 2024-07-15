# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:16:12 2024

@author: joseangelperez
"""

import numpy as np
import matplotlib.pyplot as plt

from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

# Definimos la funcion
def f(x):
    return .25*np.square(x) + 1*np.random.randn(x.size)

# Generamos los valores de X desde -5 hasta +5
# Generamos los valores de y
x=np.arange(-5,5,0.1)
y=f(x)

# Generamos nuestro arbol de regresion
arbol_reg=DecisionTreeRegressor(max_depth=2)
x_re=x.reshape(-1,1)
arbol_reg.fit(x_re,y)
arbol_data=export_graphviz(arbol_reg, out_file=None,rounded=True, filled=True)

graph=graph_from_dot_data(arbol_data)
graph.write_png("arbol_regresion.png")

xl=np.arange(-5,5,0.1)
yl=arbol_reg.predict(xl.reshape(-1,1))

#representamos graficamente los datos
plt.plot(x,y,marker='x')
plt.plot(xl,yl,marker='+')
 