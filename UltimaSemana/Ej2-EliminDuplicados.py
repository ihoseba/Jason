# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:41:48 2024

@author: joseangelperez

Ejercicio 2: Eliminación de duplicados en una lista
Escribe un programa que permita al usuario ingresar una lista de números.
El programa debe eliminar cualquier número duplicado de la lista
y mostrar el resultado.
"""

# Ejemplo
# lista=[1,2,3,4,3,2,1,2,3,4,5,6]
entrada=input("Dame una lista de numeros separados por comas: ")

lista= [int(x) for x in entrada.split(",")]
1,23
listaCompactada=[]
for i in lista:
    print(i)
    if i not in listaCompactada:
        listaCompactada.append(i)
    
print(lista)
print(listaCompactada)

