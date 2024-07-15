""" Fichero funcion de contar vocales """
# Crea una función que reciba una cadena de texto y devuelva el número de vocales que contiene.
vocales=['a','e','i','o','u','A','E','I','O','U']

def contar_vocales(cadena):
    """ funcion para contar vocales """
    n=0
    for car in cadena:
        if car in vocales:
            n+=1
    return n

# Prueba la función con al menos 3 cadenas distintas

CADENA_PRUEBA="aeiouqwert"
print(f'La Cadena {CADENA_PRUEBA} contiene {contar_vocales(CADENA_PRUEBA)}')
CADENA_PRUEBA="aeiouaeiouaeiouqwert"
print(CADENA_PRUEBA)
print(contar_vocales(CADENA_PRUEBA))
CADENA_PRUEBA="en un lugar de la mancha"
print(CADENA_PRUEBA)
print(contar_vocales(CADENA_PRUEBA))
