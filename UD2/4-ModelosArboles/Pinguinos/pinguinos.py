# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:11:57 2024

@author: joseangelperez

Actividad práctica: Árbol de clasificación con penguins

Haciendo uso del dataset penguins_mod.csv. Genera un nuevo árbol de 
clasificación, buscando el ajuste que mejor grado de impureza devuelva.

"""

# Cargar dataset y mostrar .head() del dataframe

from seaborn import lmplot, pairplot
from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

from PIL import Image

# Cargamos el fichero
#species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
file_path = 'penguins_mod.csv'
penguins = pd.read_csv(file_path,usecols=['bill_length_mm', 'bill_depth_mm',
                                          'flipper_length_mm','body_mass_g'])
#Visualizar primeras lineas
print(penguins.head(5))

penguins_datos=pd.DataFrame()
penguins_datos['bill_length_mm']=penguins['bill_length_mm']
penguins_datos['bill_depth_mm']=penguins['bill_depth_mm']
penguins_datos['flipper_length_mm']=penguins['flipper_length_mm']
penguins_datos['body_mass_g']=penguins['body_mass_g']
column_names = ['bill_length_mm', 'bill_depth_mm','flipper_length_mm','body_mass_g']
target = penguins.values[1::2, 2]


X=penguins_datos
#X = penguins_datos.DataFrame(penguins_datos.c_[df['bill_length_mm'], df['bill_depth_mm'],
#                        df['flipper_length_mm'],df['body_mass_g']],
#                 columns=['bill_length_mm', 'bill_depth_mm',
#                          'flipper_length_mm','body_mass_g'])
y = penguins['species']

pass








"""
# Obtener diagrama de correlación entre las variables bill_length_mm y
# bill_depth_mm Considerando como variable de discriminación 'species'
lmplot(penguins, x='flipper_length_mm', y='body_mass_g',
       hue='species', fit_reg=False)
plt.grid()

lmplot(penguins, x='bill_length_mm', y='bill_depth_mm',
       hue='species', fit_reg=False)
plt.grid()

# Dibujar la matriz de correlaciones entre todas las variables existentes.
# Discriminando de nuevo por 'species'
pairplot(penguins, hue='island')

# Dibujar la matriz de correlaciones entre todas las variables existentes.
# Discriminando de nuevo por 'species'
pairplot(penguins, hue='species')
"""

# Extraer y formatear los datos 
#X=penguins["bill_length_mm"]+penguins["bill_depth_mm"]
X = penguins.DataFrame(np.c_[df['bill_length_mm'], df['bill_depth_mm']],
                       columns=['bill_length_mm', 'bill_depth_mm'])
X=penguins["bill_length_mm"]
y=penguins["species"]

# Hacer Split de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)



#Generar  Arbol de Decision
n=int(0.2 * len(X))
arbol_reg=DecisionTreeClassifier(criterion="squared_error",max_leaf_nodes=n)
x_re=X.reshape(-1,1)
arbol_reg.fit(x_re,y)
arbol_data=export_graphviz(arbol_reg, out_file=None,rounded=True, filled=True)

graph=graph_from_dot_data(arbol_data)
graph.write_png("arbol_regresion_penguin.png")

# Abre la imagen
img = Image.open("arbol_regresion_penguin.png")
# Muestra la imagen en una ventana
img.show()

xl=np.arange(-5,5,0.1)
yl=arbol_reg.predict(xl.reshape(-1,1))

#representamos graficamente los datos
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X, y,marker='x')
ax.plot(xl, yl,marker='+')
ax.set_title("Gráfica")

plt.plot(X,y,marker='x')
plt.plot(xl,yl,marker='+')

