# Ordenar la lista de menor a mayor
lista = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
version_ordenar_lista=[1,1,1]

def ordenar_lista():
    # Imprime la lista original y la lista ordenada por separado
    print(f'Lista Original: {lista}')
    lista.sort()
    print(f'Lista Ordenada: {lista}')
