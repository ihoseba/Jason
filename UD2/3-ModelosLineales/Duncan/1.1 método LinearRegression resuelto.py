# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:38:49 2024

@author: MarkelP
"""

#Importamos los módulos y librerías necesarios
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#Cargamos el fichero de datos (recuerda particularizar la ruta en función de donde hayas guardado el fichero .csv)
nomb_fich='FtG_0_7s0tx4nn15-Duncan.csv'

with open(nomb_fich) as f:
     lectura=csv.reader(f,delimiter=',')
     encabezados=next(lectura)
     ingresos=[]
     educacion=[]
     prestigio=[]
     for fila in lectura:
         ing=int(fila[2])
         ingresos.append(ing)

         educ=int(fila[3])
         educacion.append(educ)
         prest=int(fila[4])
         prestigio.append(prest)

print("Estructura de la variable ingresos:",type(ingresos),"Tamaño:",len(ingresos))
print("Estructura de la variable educacion:",type(educacion),"Tamaño:",len(educacion))
print("Estructura de la variable prestigio:",type(prestigio),"Tamaño:",len(prestigio))


#Como income es la variable independiente, la transformamos en un array con 45 filas y 1 columna

#Para ello, primero tienes que crear el array en sí y luego transponerlo.
#Por su parte, prestige es la variable independiente, por lo que solo bastará con crear el correspondiente array de NumPy. 

ingresos=np.array([ingresos])
ingresos=np.transpose(ingresos)
ingresos2=np.array(ingresos) # Parece hacer lo mismo
prestigio=np.array(prestigio)

#Generamos los conjuntos de entrada y de entrenamiento. Invocamos el procedimiento de regresión.
X_train, X_test, y_train, y_test = train_test_split(ingresos,prestigio,random_state=20)
lr=LinearRegression().fit(X_train,y_train)
y_pred=lr.predict(X_test)

#Valor del error cuadrático medio redondeado
print("Error cuadrático medio:",round(mean_squared_error(y_test, y_pred),2))
 
#Valores de los coeficientes de determinación para cada conjunto
print("Valor del coeficiente de determinación del conjunto de entrenamiento:",
      round(lr.score(X_train,y_train),3))
print("Valor del coeficiente de determinación del conjunto de prueba:",
      round(lr.score(X_test,y_test),3))
#Salidas gráficas

plt.scatter(X_test, y_test, color='black')

plt.xlabel("Ingresos")
plt.ylabel("Prestigio")
plt.plot(X_test, y_pred, color='blue', linewidth=1)
