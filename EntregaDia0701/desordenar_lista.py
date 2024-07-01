""" barajar una lista, desordenarla """

import random
import os

comidas_mediterraneas = [
    "Fideuá",
    "Hummus",
    "Tabulé",
    "Baba ganoush",
    "Tzatziki",
    "Moussaka",
    "Kebabs",
    "Falafel",
    "Taramosalata",
    "Ratatouille",
    "Couscous",
    "Focaccia",
    "Tiramisú",
    "Baklava",
    # Agrega más elementos aquí si lo deseas
]

os.system('cls')
#Main
print (comidas_mediterraneas)
n=0
while True:
    random.shuffle(comidas_mediterraneas)
    n+=1
    print (f'Iteracion {n} de lista aleatoria')
    print (comidas_mediterraneas)
    salir=input("Continuar? (Pulsa Enter) / Salir (x)")
    if salir.lower() == 'x':
        break

