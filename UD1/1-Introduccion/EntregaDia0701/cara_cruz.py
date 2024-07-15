""" Script que devuelva Cara o cruz eligiendo numero de veces y se quede ultimo """
import random
import os

CARA_CRUZ=['CARA','CRUZ']

def cara_cruz():
    os.system('cls')
    #Main
    while True:
        num=input("Dame el numero de iteraciones:")
        for i in num:
            seleccion = random.choice(CARA_CRUZ)
        print("Ha salido: ", seleccion)
        salir=input("Continuar? (Pulsa Enter) / Salir (x)")
        if salir == 'x':
            break

cara_cruz()
