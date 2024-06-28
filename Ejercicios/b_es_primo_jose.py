""" Dichero par comprobar si un numero es primo """
# Escribe una función que verifique si un número es primo.
def es_primo(n):
    """Comprobar si primo"""
    for i in range(2, n):
        if (n % i) == 0:
            return False
    return True

# Prueba la función con varios números

lista_pruebas=[1,2,3,4,5,6,7,8,9]

for numero in lista_pruebas:
    if es_primo(numero):
        print(f'Si, {numero} es primo')
    else:
        print(f'No, {numero} no es primo')

# cambio para Git
