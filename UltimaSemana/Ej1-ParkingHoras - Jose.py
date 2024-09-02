# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:34:59 2024

@author: joseangelperez

Ejercicio 1: Parking por horas
Supón que estás desarrollando una aplicación para un estacionamiento.
 Las tarifas establecidas, en función del tiempo de permanencia, son:

    • Las personas que estacionan menos de 1 hora no pagan nada.
    • Las personas que estacionan entre 1 y 3 horas pagan 2 € por hora.
    • Las personas que estacionan entre 3 y 5 horas pagan 1,5 € por hora.
    • Las personas que estacionan más de 5 horas pagan una tarifa fija de 10 €.

Genera un código en Python que pregunte cuántas horas ha permanecido el
 vehículo en el estacionamiento y que devuelva por pantalla cuánto debe pagar.
"""

horas=input("Cuantas horas has estacionado?")
horas=float(horas)

if horas < 1:
    precioXhora = 0
    precioTotal = horas * precioXhora
elif horas <3:
    precioXhora=2;
    precioTotal = horas * precioXhora
elif horas <5:
    precioXhora=1.5;
    precioTotal = horas * precioXhora
elif horas >5:
    precioXhora="Fijo 10 €";
    precioTotal=10

print("Has estacionado ",horas, " horas")
print("El precio por hora es de ",precioXhora)
print("Has de pagar ",precioTotal, " Euros")
    
