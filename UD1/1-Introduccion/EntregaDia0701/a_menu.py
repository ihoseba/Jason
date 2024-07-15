""" Todos los ejercicios de hoy"""
import cara_cruz

opciones:['c','d']
#Main
while True:
    opcion=input(f'selecciona una opcion: ')
    if opcion == 'c':
        cara_cruz.cara_cruz()
    elif opcion == 'x':
        break
    else:
        continue
