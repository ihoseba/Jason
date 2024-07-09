# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:33:03 2024

@author: joseangelperez

Imagina que trabajas en una agencia de medios y te han encargado que estudies 
el prestigio conferido por la sociedad respecto a ciertas profesiones.
Para ello, vamos a suponer que los datos recogidos en el dataset Duncan
(correspondientes a una investigación realizada en 1961 y muy empleados en el 
ámbito educativo) continúan vigentes en la actualidad. 
En la sección «Descargables» dispones de la hoja de cálculo Duncan.csv,
con dichos datos separados por comas. Cuando procedas a su descarga comprobarás
que el dataset contiene 5 columnas: 
• En la primera se relacionan las distintas profesiones a analizar
 (150 en total). 
• En la segunda se indica en qué tipología se encuadra la profesión:
 o Blue collar, que es el término inglés que hace referencia a un trabajo de
 naturaleza manual. 
 o White collar, que caracteriza labores administrativas y comerciales. 
 o Prof, relativo a trabajos profesionales o gerenciales. 
• La tercera columna (income o ingresos) recoge el porcentaje de personas,
 con la profesión correspondiente a la fila en la que se relacionan los datos,
 que perciben un salario superior a los 32 000 € anuales de la época actual. 
• La cuarta columna (education o educación) muestra el porcentaje de personas
 trabajadoras con la profesión que corresponda que cuentan con un grado
 universitario. 
• La quinta columna (prestige o prestigio) contiene los porcentajes de
 respuesta de personas que participaron en una encuesta y que calificaron a
 la profesión de la que se trate como buena o idónea. 
"""

import pandas as pd
import csv
import matplotlib.pyplot as plt

file_path = "FtG_0_7s0tx4nn15-Duncan.csv"


# En base a lo expuesto: 
# 1. Descarga los datos de la hoja de cálculo y averigua su estructura, con 
# objeto de identificar las transformaciones que permitan aplicar el método
# fit. Denota las variables income, education y prestige como ingresos. 


with open(file_path) as f:
    lectura = csv.reader(f, delimiter=',')
    encabezados = next(lectura)
    locualo = []
    tipo = []
    income = []
    education = []
    prestige = []
    for fila in lectura:
        locualo.append(fila[0])
        tipo.append(fila[1])
        income.append(fila[2])
        education.append(fila[3])
        prestige.append(fila[4])

# probemos a plotear

plt.plot(X,y,'o')
plt.ylim(-3,3) 
plt.xlabel('Variable independiente') 
plt.ylabel('Variable dependiente')

# 2. Analiza la regresión lineal entre income (como variable independiente)
# y prestige (como variable independiente). En concreto, debes obtener el
# error cuadrático medio del conjunto de prueba, los coeficientes de regresión
# para ambos conjuntos y una representación gráfica del modelo sobre el
# conjunto de prueba. ¿Cómo valoras los resultados obtenidos? Interprétalos. 
