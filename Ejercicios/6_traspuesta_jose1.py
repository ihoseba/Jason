# Escribe una función que calcule la transpuesta de una matriz.
from copy import deepcopy
# Prueba la función con una matriz

# Para una entrada como esta:
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Debería devolver:
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

matriz=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# dimensiones x e y han de ser iguales

matriz_copia = deepcopy(matriz)

if len(matriz)==len(matriz[0]):
    #comprobar consistencia
    m=len(matriz)
    for x in range(0,m):
        for y in range(0,m):
            print(matriz[x][y])
            # if m!=len(matriz[x][y]):
            #     #Inconsistente
            #     print(f'Error en dimension {x} , {y}')
else:
    print("Error, matriz no es cuadrada")

x = y = m
matriz_traspuesta=deepcopy(matriz)

for x in range (0,m):
    for y in range(0,m):
        matriz_traspuesta[x][y]=matriz[y][x]
        print(matriz[x])

print(matriz)
print(matriz_traspuesta)
print(len(matriz))
print(len(matriz[0]))

