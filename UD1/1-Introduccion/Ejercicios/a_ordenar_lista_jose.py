"""Ordenar la lista de menor a mayor"""
# Ordenar la lista de menor a mayor
lista = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
version_ordenar_lista=[1,1,1]

# funcion para ordenar la lista
def ordenar_lista():
    """ para ordenar lista"""
    # Imprime la lista original y la lista ordenada por separado
    print(f'Lista Original: {lista}')
    lista.sort()
    print(f'Lista Ordenada: {lista}')
