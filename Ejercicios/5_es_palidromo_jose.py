# Escribe una función que verifique si una cadena es un palíndromo.
def es_palindromo(cadena):
    return cadena==cadena[::-1] 
# modificamos desarrollo

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

