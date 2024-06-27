# Crea una función que reciba una cadena de texto y devuelva el número de vocales que contiene.
vocales=['a','e','i','o','u','A','E','I','O','U']

def contar_vocales(cadena):
    n=0
    for car in cadena:
        if car in vocales:
            n+=1 
    return n

# Prueba la función con al menos 3 cadenas distintas

cadena_prueba="aeiouqwert"
print(f'La Cadena {cadena_prueba} contiene {contar_vocales(cadena_prueba)}')
cadena_prueba="aeiouaeiouaeiouqwert"
print(cadena_prueba)
print(contar_vocales(cadena_prueba))
cadena_prueba="en un lugar de la mancha"
print(cadena_prueba)
print(contar_vocales(cadena_prueba))

