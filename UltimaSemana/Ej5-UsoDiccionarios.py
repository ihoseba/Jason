# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:17:47 2024

@author: joseangelperez

Ejercicio 4: Uso de Diccionarios
Crea una función contar_ocurrencias que reciba una lista de palabras y
devuelva un diccionario donde las llaves sean las palabras y los valores
sean el número de veces que cada palabra aparece en la lista.

"""

def contar_ocurrencias(lista_palabras):
    # Crear un diccionario vacío para almacenar las ocurrencias
    dic_palabras = {}
    
    # Recorrer la lista de palabras
    for palabra in lista_palabras:
        # Si la palabra ya está en el diccionario, incrementar su contador
        if palabra in dic_palabras:
            dic_palabras[palabra] += 1
        # Si la palabra no está en el diccionario, agregarla con un contador de 1
        else:
            dic_palabras[palabra] = 1
    
    return dic_palabras


# Solicitar al usuario que ingrese una lista de palabras separadas por comas
entrada = input("Ingresa una lista de palabras separadas por comas: ")
# ejemplo
# entrada="casa,Juan, si,bocadillo ,si,Juan"
# Convertir la entrada en una lista de palabras
lista_palabras = [palabra.strip() for palabra in entrada.split(",")]

# ejemplo
# lista_palabras=["casa","Juan","si","bocadillo","si","Juan"]

# Contar las ocurrencias de cada palabra
resultado = contar_ocurrencias(lista_palabras)

# Mostrar el resultado
print("Ocurrencias de cada palabra:", resultado)