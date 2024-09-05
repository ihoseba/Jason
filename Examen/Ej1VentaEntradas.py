# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:31:36 2024

@author: joseangelperez
Edad <5 precio 0
Edad =5 ~ =12  precio 5
Edad =13 ~ =60  precio 15
Edad >60, Precio 8


"""

edad=input("Que edad tienes?: ")
edad=int(edad)

if edad > -1 and edad <5: # Menor de 5
    precio=0
elif edad >4 and edad <13: # Entre 5 y 12
    precio=5
elif edad >13 and edad <61: # Entre 13 y 60
    precio=15
else: # Mayor de 60
    precio=8

if edad>-1:
    print(" Has de pagar ", precio, " euros")
    print(" porque tienes ", edad, " años")
else:
    print("No has nacido aún?")