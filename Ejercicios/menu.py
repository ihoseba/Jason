""" Menu incluyendo todas las opciones """
import os
import a_ordenar_lista_jose as ord_list
from b_es_primo_jose import es_primo
from c_contar_caracteres_jose import contar_caracteres
import d_contar_vocales_jose
from e_es_palidromo_jose import es_palindromo
from f_traspuesta_jose import trasponer
from g_fizbuzz_jose import fizz_buzz

# Hacer un menu para entrar en los diferentes ejercicios
menu=[
      '---------------------',
      '1, Ordenar Lista',
      '2, Es Primo',
      '3, Contar Caracteres',
      '4, Contar Vocales',
      '5, Es Palindromo',
      '6, Trasponer',
      '7, Fizz Buzz',
      '---',
      '0, Salir',
      '---------------------',
      ]

OPCIONES=['0','1','2','3','4','5','6','7']

# Selecionar una opcion
def seleccion_opcion(sel_menu):
    """ para seleccionar una opcion """
    for texto in sel_menu:
        print(texto)
    sel_opcion=input("Opcion ?")
    return sel_opcion

def none():
    """ None is none"""
    return

# Main
while True:
    os.system('cls')
    opcion=seleccion_opcion(menu)
    if opcion in OPCIONES:
        print(f'Opcion seleccionada \"{menu[int(opcion)]}\"')
    else:
        input('Error en la opcion seleccionada, pulse enter para continuar')
        continue
    if opcion=='1':
        ord_list.ordenar_lista()
    elif opcion=='2':
        num=int(input("dame un numero: "))
        if es_primo(num) is True:
            print(' Sip, primo es')
        else:
            print('No, no es primo')
    elif opcion=='3':
        CADENA='pruebapru'
        print(f'ejemplo #2 {CADENA}')
        print(contar_caracteres(CADENA))
        CADENA='en un lugar de la mancha'
        print(f'ejemplo #1 {CADENA}')
        print(contar_caracteres(CADENA))
    elif opcion=='4':
        CADENA_PRUEBA="aeiouqwert"
        print(f'La Cadena {CADENA_PRUEBA} contiene '
              f'{d_contar_vocales_jose.contar_vocales(CADENA_PRUEBA)}')
        CADENA_PRUEBA="aeiouaeiouaeiouqwert"
        print(CADENA_PRUEBA)
        print(d_contar_vocales_jose.contar_vocales(CADENA_PRUEBA))
        CADENA_PRUEBA="en un lugar de la mancha"
        print(CADENA_PRUEBA)
        print(d_contar_vocales_jose.contar_vocales(CADENA_PRUEBA))
    elif opcion=='5':
        cadenas=[
            "bocadillo",
            "reconocer",
            "aeihiea"]
        for cadena in cadenas:
            if es_palindromo(cadena):
                print(f'Si, {cadena} es palindromo')
            else:
                print(f'pues no, {cadena} no es palíndromo')
    elif opcion=='6':
        trasponer()
    elif opcion=='7':
        for n in range(1,100+1):
            fizz_buzz(n)
    elif opcion=='0':
        break
    else:
        break
    input('Seguir?')
    continue

# Lanzar funciones de los otros ficheros
#
