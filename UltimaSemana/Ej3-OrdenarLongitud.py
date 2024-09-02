# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:55:33 2024

@author: joseangelperez

Ejercicio 3: Ordenar por longitud
Crea un programa que permita al usuario ingresar una lista de palabras.
El programa debe ordenar y mostrar las palabras según su longitud,
de la más corta a la más larga

"""

palabras=["bocadillo","jamon","mesa"]

def ordenar_por_longitud(lista_palabras):
    # Ordenar la lista de palabras por su longitud
    lista_ordenada = sorted(lista_palabras, key=len)
    return lista_ordenada

# Solicitar al usuario que ingrese una lista de palabras separadas por comas
entrada = input("Ingresa una lista de palabras separadas por comas: ")
# Convertir la entrada en una lista de palabras
lista_palabras = [palabra.strip() for palabra in entrada.split(",")]

print(lista_palabras)

# Ordenar las palabras por longitud
resultado = ordenar_por_longitud(lista_palabras)

# Mostrar el resultado
print("Lista de palabras ordenadas por longitud:", resultado)
