import os
import a_ordenar_lista_jose as ord_list
from b_es_primo_jose import es_primo as es_primo

# Hacer un menu para entrar en los diferentes ejercicios
menu=['1, Ordenar Lista',
      '2, Es Primo',
      '3, Contar Caracteres',
      '4, Contar Vocales',
      'x, ...',
      '0, Salir'
      ]

# Selecionar una opcion
def seleccion_opcion(menu):
    for texto in menu:
        print(texto)
    opcion=input("Opcion ?")
    return opcion

# Main
while True:
    os.system('cls')
    opcion=seleccion_opcion(menu)
    if opcion=='1':
        print(f'Opcion seleccionada \"{menu[int(opcion)-1]}\"')
        ord_list.ordenar_lista()
    elif opcion=='2':
        print(f'Opcion seleccionada \"{menu[int(opcion)-1]}\"')
        num=int(input("dame un numero: "))
        if es_primo(num) == True:
            print(' Sip, primo es')
        else:
            print('No, no es primo')
    elif opcion=='3':
        print(f'Opcion seleccionada \"{menu[int(opcion)-1]}\"')
    elif opcion=='4':
        print(f'Opcion seleccionada \"{menu[int(opcion)-1]}\"')
    else:
        break
    input('Seguir?')
    continue

# Lanzar funciones de los otros ficheros
# v