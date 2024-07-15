# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:16:12 2024

@author: joseangelperez

Actividad práctica: limitar el proceso de sobreajuste 

Los procedimientos de regularización también se pueden emplear en regresión
para limitar el proceso de sobreajuste. Para ello, vas a realizar lo siguiente:
 
    a)  En primer lugar, entrena el modelo anterior sin ningún tipo de
    limitación.
    b)  Seguidamente, impón que el número máximo de nodos finales sea el 20 %
    del tamaño de la muestra de puntos.
    c)  Para finalizar, obtén las gráficas predictivas en ambas situaciones
    y compara dichos resultados.

Una vez realizadas todas las cuestiones, comprueba tu respuesta.
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

# a)  En primer lugar, entrena el modelo anterior sin ningún tipo de
# limitación.
# Generamos nuestro arbol de regresion
arbol_reg1=DecisionTreeRegressor()
x_re1=x.reshape(-1,1)
arbol_reg1.fit(x_re1,y)
arbol_data1=export_graphviz(arbol_reg1, out_file=None,rounded=True, filled=True)

graph1=graph_from_dot_data(arbol_data1)
graph1.write_png("arbol_regresion1.png")

xl1=np.arange(-5,5,0.1)
yl1=arbol_reg1.predict(xl1.reshape(-1,1))

#representamos graficamente los datos
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, y,marker='x')
ax1.plot(xl1, yl1,marker='+')
ax1.set_title("Gráfica 1")

plt.plot(x,y,marker='x')
plt.plot(xl1,yl1,marker='+')

# b)  Seguidamente, impón que el número máximo de nodos finales sea el 20 %
# del tamaño de la muestra de puntos.
n=int(0.2 * len(x))
arbol_reg2=DecisionTreeRegressor(min_samples_leaf=n)
x_re2=x.reshape(-1,1)
arbol_reg2.fit(x_re2,y)
arbol_data2=export_graphviz(arbol_reg2, out_file=None,rounded=True, filled=True)

graph2=graph_from_dot_data(arbol_data2)
graph2.write_png("arbol_regresion2.png")

xl2=np.arange(-5,5,0.1)
yl2=arbol_reg2.predict(xl2.reshape(-1,1))

#representamos graficamente los datos
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, y,marker='x')
ax2.plot(xl2, yl2,marker='+')
ax2.set_title("Gráfica 2")

plt.plot(x,y,marker='x')
plt.plot(xl2,yl2,marker='+')

 

# a)  En primer lugar, entrena el modelo anterior sin ningún tipo de
# limitación.
# b)  Seguidamente, impón que el número máximo de nodos finales sea el 20 %
# del tamaño de la muestra de puntos.
# c)  Para finalizar, obtén las gráficas predictivas en ambas situaciones
# y compara dichos resultados.


