# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:05:30 2024

@author: joseangelperez

Ejercicio 4: Manipulación de Listas
Crea una función procesar_lista que reciba una lista de números enteros
y devuelva una nueva lista que contenga solo los números pares de la 
lista original. La función debe manejar listas vacías y listas que no 
contengan números pares.

"""

def nums_pares(num_enteros):
    lista_pares = []
    for i in num_enteros:
        if i%2 == 0:
            lista_pares.append(i)

    return(lista_pares)

enteros=input("Dame un lista de enteros separados por comas: ")
lista_enteros=[int(entero) for entero in enteros.split(",")]

print(lista_enteros)

lista_pares=nums_pares(lista_enteros)

print(lista_pares)

