"""
Caso pr谩ctico: coeficientes de Kendall y de Spearman 

Sup贸n que trabajas en una consultora a la que le han asignado la tarea de
 analizar la valoraci贸n de mil productos que comercializa una empresa a 
 trav茅s de comercio electr贸nico. Para ello, supondremos que aplicamos un 
 algoritmo de Machine Learning que permite valorar cada uno de esos productos 
 en una escala de 1 a 10 (siendo 1 la peor valoraci贸n y 10 la mejor). 
 A fin de disponer de un resultado con el que trabajar genera, 
 en primer lugar, una lista de 1000 n煤meros aleatorios enteros, 
 con valores comprendidos entre 1 y 10. 
驴Qu茅 suceder谩 si trabajamos con este mismo listado, pero con una 
escala diferente? 驴Cambiar谩 la informaci贸n contenida en la misma? 
Genera para ello una segunda lista multiplicando cada uno de los t茅rminos 
de la primera lista por 5 y sumando 8 al valor de cada resultado.

1. Analiza la correlaci贸n existente entre ambas listas mediante los 
coeficientes de Kendall y de Spearman. 
Recuerda la idoneidad de emplear una semilla antes de generar los n煤meros 
aleatorios, mediante la funci贸n random.seed() -debes introducir un 
n煤mero entero entre los par茅ntesis- y que existe una funci贸n como 
random.randint(x,y) para acotar el intervalo de generaci贸n. 

2. Genera, a continuaci贸n, otras dos listas, cada una con 1000 
t茅rminos, ambas aleatorias, con n煤meros entre el 1 y el 10, pero con 
semillas distintas. 
Determina la correlaci贸n existente entre ambas listas con los mismos 
coeficientes que en el primer apartado. 

3. Analiza los resultados obtenidos en el primer y segundo apartado. 
驴Crees que se pod铆an esperar a priori?
"""

"""
Copilot, haz codigo python que genere una lista de n=1000 numeros aleatorios
enteros, con valores comprendidos entre 1 y 10. 
Genera una segunda lista multiplicando cada uno de los terminos de la primera 
lista por 5 y sumando 8 al valor de cada resultado.
Analiza la correlaci贸n existente entre ambas listas mediante los 
coeficientes de Kendall y de Spearman. imprimiendo los resultados
Genera, a continuaci贸n, otras dos listas, cada una con 1000 
t閞rminos, ambas aleatorias, con n鷐eros entre el 1 y el 10, pero con 
semillas distintas. 
Determina la correlaci髇 existente entre ambas listas con los mismos 
coeficientes que en el primer apartado. 
"""

import random
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr

# Configuraci髇 de la semilla para reproducibilidad
random.seed(42)  # Puedes cambiar la semilla si lo deseas

# Generar la primera lista
lista1 = [random.randint(1, 10) for _ in range(1000)]
lista1_transformada = [(x * 5) + 8 for x in lista1]

# Calcular coeficientes de correlaci髇
kendall_corr, k_p = kendalltau(lista1, lista1_transformada)
spearman_corr, s_p = spearmanr(lista1, lista1_transformada)
# Imprimir resultados
print("-----","Numeros aleatorios Desplazados x5 + 8")
print(f"Coeficiente de Kendall: {kendall_corr:.4f}")
print(f"Probabilidad: {k_p:.4f}")
print(f"Coeficiente de Spearman: {spearman_corr:.4f}")
print(f"Probabilidad: {s_p:.4f}")


# Generar la segunda lista con semilla diferente
random.seed(123)  # Cambia la semilla si prefieres otra
lista2 = [random.randint(1, 10) for _ in range(1000)]

# Calcular coeficientes de correlaci髇
kendall_corr, k_p = kendalltau(lista1, lista2)
spearman_corr, s_p = spearmanr(lista1, lista2)

# Imprimir resultados
print("-----","Otros Numeros Aleatorios","-----")
print(f"Coeficiente de Kendall: {kendall_corr:.4f}")
print(f"Probabilidad: {k_p:.4f}")
print(f"Coeficiente de Spearman: {spearman_corr:.4f}")
print(f"Probabilidad: {s_p:.4f}")

# Generar Otra lista con una relacion desplazando n elementos 
desp=345
def desplazar(lista, n):
    return lista[-n:] + lista[:-n]

lista3 = desplazar(lista1, desp)

# Calcular coeficientes de correlaci髇
kendall_corr, k_p = kendalltau(lista1, lista3)
spearman_corr, s_p = spearmanr(lista1, lista3)

# Imprimir resultados
print(f"----- Lista desplazada {desp} posiciones -----")
print(f"Coeficiente de Kendall: {kendall_corr:.4f}")
print(f"Probabilidad: {k_p:.4f}")
print(f"Coeficiente de Spearman: {spearman_corr:.4f}")
print(f"Probabilidad: {s_p:.4f}")
