# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:12:49 2024

@author: joseangelperez
"""

# Actividad práctica: realización de ajustes
# A fin de que puedas comprender, de una manera práctica, los condicionantes
# del overfitting y del underfitting, te propongo que retomes un conjunto de
# datos sobre los que ya has trabajado al analizar el ajuste polinómico:
# se trata de los recogidos en Raschka y Mirjalili (2018), correspondientes
# (258.0, 236.4), (270.0, 234.4), (294.0, 252.8), (320.0, 298.6), (342.0, 314,2),
# (368.0, 342.2), (396.0, 360.8), (446.0, 368.0), (480.0, 391.2) y (586.0, 390.8).

# En base a lo ya estudiado, ajusta dichos datos considerando:
#    1. Un valor constante (es decir, un ajuste tipo y = w0).
#    2. Un ajuste lineal.
#    3. Un ajuste cuadrático.
#    4. Un ajuste polinómico con g = 4.
#    5. Un ajuste polinómico con g = 9.

# Determina, para cada ajuste, el error cuadrático medio y el valor del
# coeficiente de determinación.
# Realiza todas las cuestiones expuestas y, posteriormente, comprueba tu
# respuesta.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#Cargamos los datos directamente
X = np.array([258.0,270.0,294.0,320.0,342.0,368.0,396.0,446.0,480.0,586.0])
y = np.array([236.4,234.4,252.8,298.6,314.2,342.2,360.8,368.0,391.2,390.8])
plt.scatter(X,y)
plt.xlabel("Variable x")
plt.ylabel("Variable y")

# Preparamos previamente X para que sea un array columna
X = np.transpose([X])
lr = LinearRegression().fit(X,y)
X_lin_ajuste = np.arange(250,600,10)
X_lin_ajuste = np.transpose([X_lin_ajuste])
y_lin_ajuste = lr.predict(X_lin_ajuste)

"""
#    1. Un valor constante (es decir, un ajuste tipo y = w0).
# ++++++++++++++++
#    2. Un ajuste lineal.
# ++++++++++++++++
#Ajuste lineal.

#Dibujamos el ajuste lineal. Empleamos las líneas discontinuas para representarlo.
#plt.plot(X_lin_ajuste,y_lin_ajuste,label = "Ajuste lineal",linestyle = '--',color = 'green')

y_lin_pred = lr.predict(X)
print(" Error cuadrático medio del modelo lineal:\t\t\t\t\t",
      round(mean_squared_error(y,y_lin_pred),4))
print("Coeficiente de determinación del modelo lineal:",
      round(r2_score(y,y_lin_pred),4))

#Dibujamos
plt.plot(X,y_lin_pred,label = "Ajuste polinómico",color = 'yellow')

# ++++++++++++++++
#    3. Un ajuste cuadrático.
# ++++++++++++++++
#Ajuste polinómico
cuadratico = PolynomialFeatures(degree = 2)
X_cuadr = cuadratico.fit_transform(X)
pr = LinearRegression().fit(X_cuadr,y)
y_cuadr_ajuste = pr.predict(cuadratico.fit_transform(X_lin_ajuste))

#Predicciones: errores y coeficientes de determinación
y_cuadr_pred = pr.predict(X_cuadr)

print(" Error cuadrático medio del modelo polinómico g=2:\t\t\t\t\t",
      round(mean_squared_error(y,y_cuadr_pred),4))
print("Coeficiente de determinación del modelo polinómico g=2:",
      round(r2_score(y,y_cuadr_pred),4))

#Dibujamos el ajuste cuadrático
plt.plot(X_lin_ajuste,y_cuadr_ajuste,label = "Ajuste cuadrático 2",color = 'red')

# ++++++++++++++++
#    4. Un ajuste polinómico con g = 4.
# ++++++++++++++++
#Ajuste polinómico
cuadratico_4 = PolynomialFeatures(degree = 4)
X_cuadr_4 = cuadratico_4.fit_transform(X)
pr_4 = LinearRegression().fit(X_cuadr_4,y)
y_cuadr_ajuste_4 = pr_4.predict(cuadratico_4.fit_transform(X_lin_ajuste))

#Predicciones: errores y coeficientes de determinación
y_cuadr_pred_4 = pr_4.predict(X_cuadr_4)

print(" Error cuadrático medio del modelo polinómico g=4:\t\t\t\t\t",
      round(mean_squared_error(y,y_cuadr_pred_4),4))
print("Coeficiente de determinación del modelo polinómico g=4:",
      round(r2_score(y,y_cuadr_pred_4),4))

#Dibujamos el ajuste cuadrático
plt.plot(X_lin_ajuste,y_cuadr_ajuste_4,label = "Ajuste cuadrático 4",color = 'blue')

# ++++++++++++++++
#    5. Un ajuste polinómico con g = 9.
# ++++++++++++++++
cuadratico_9 = PolynomialFeatures(degree = 9)
X_cuadr_9 = cuadratico_9.fit_transform(X)
pr_9 = LinearRegression().fit(X_cuadr_9,y)
y_cuadr_ajuste_9 = pr_9.predict(cuadratico_9.fit_transform(X_lin_ajuste))

#Predicciones: errores y coeficientes de determinación
y_cuadr_pred_9 = pr_9.predict(X_cuadr_9)

print(" Error cuadrático medio del modelo polinómico g=9:\t\t\t\t\t",
      round(mean_squared_error(y,y_cuadr_pred_9),4))
print("Coeficiente de determinación del modelo polinómico g=9:",
      round(r2_score(y,y_cuadr_pred_9),4))

#Dibujamos el ajuste cuadrático
plt.plot(X_lin_ajuste,y_cuadr_ajuste_9,label = "Ajuste cuadrático 9",color = 'green')

# ++++++++++++++++
#    n. Un ajuste polinómico con g = n.
# ++++++++++++++++
n=0
cuadratico_n = PolynomialFeatures(degree = n)
X_cuadr_n = cuadratico_n.fit_transform(X)
pr_n = LinearRegression().fit(X_cuadr_n,y)
y_cuadr_ajuste_n = pr_n.predict(cuadratico_n.fit_transform(X_lin_ajuste))

#Predicciones: errores y coeficientes de determinación
y_cuadr_pred_n = pr_n.predict(X_cuadr_n)

print(" Error cuadrático medio del modelo polinómico g=n:\t\t\t\t\t",
      round(mean_squared_error(y,y_cuadr_pred_n),4))
print("Coeficiente de determinación del modelo polinómico g=",n,":",
      round(r2_score(y,y_cuadr_pred_n),4))

#Dibujamos el ajuste cuadrático
plt.plot(X_lin_ajuste,y_cuadr_ajuste_n,label = "Ajuste cuadrático n",color = 'black')
"""
# ++++++++++++++++
#    barre. Un ajuste polinómico con g = rango.
# ++++++++++++++++
colores=['red','blue','green','yellow','orange','purple','pink','gray',
         'cyan','brown','black','white','turquoise','lime','silver']
for i in range(0,10):
    n=i
    cuadratico_n = PolynomialFeatures(degree = n)
    X_cuadr_n = cuadratico_n.fit_transform(X)
    pr_n = LinearRegression().fit(X_cuadr_n,y)
    y_cuadr_ajuste_n = pr_n.predict(cuadratico_n.fit_transform(X_lin_ajuste))
    
    #Predicciones: errores y coeficientes de determinación
    y_cuadr_pred_n = pr_n.predict(X_cuadr_n)
    
    Err=round(mean_squared_error(y,y_cuadr_pred_n),4)
    print(f" Error cuadrático medio del modelo polinómico g={n}:\t\t\t\t\t \
          {Err}")
    print(f"Coeficiente de determinación del modelo polinómico g={n}: \
          {round(r2_score(y,y_cuadr_pred_n),4)}")
    
    #Dibujamos el ajuste cuadrático
    plt.plot(X_lin_ajuste,y_cuadr_ajuste_n,label = f"Aj.cuad.{n}, {Err}",color = colores[i])
# ++++++++++++++++

# Colocar la leyenda en la esquina superior derecha
plt.legend(loc="upper left")

plt.show()

