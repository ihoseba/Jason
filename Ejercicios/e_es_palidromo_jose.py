""" modulo es Palindromo """
# Escribe una función que verifique si una cadena es un palíndromo.
def es_palindromo(cadena):
    """ funcion para coprobar si es palindromo """
    return cadena==cadena[::-1]

# Prueba la función con al menos 3 cadenas distintas
cadenas=[
    "bocadillo",
    "reconocer",
    "aeihiea"]

for cadena in cadenas:
    if es_palindromo(cadena):
        print(f'Si, {cadena} es palindromo')
    else:
        print(f'pues no, {cadena} no es palíndromo')
