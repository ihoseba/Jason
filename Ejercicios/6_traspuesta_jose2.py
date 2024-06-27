# Escribe una función que calcule la transpuesta de una matriz.
# Prueba la función con una matriz

# Para una entrada como esta:
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Debería devolver:
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# by copilot
# Matriz original
matriz_original = [
    [1, 2, 3, 0],
    [4, 5, 6, 0],
    [7, 8, 9, 0]
]

# Imprimir la matriz transpuesta
for fila in matriz_original:
    print(fila)

# Crear una matriz vacía para la transpuesta
filas = len(matriz_original)
columnas = len(matriz_original[0])
matriz_transpuesta = [[0] * filas for _ in range(columnas)]

# Rellenar la matriz transpuesta
for i in range(filas):
    for j in range(columnas):
        matriz_transpuesta[j][i] = matriz_original[i][j]

# Imprimir la matriz transpuesta
for fila in matriz_transpuesta:
    print(fila)

prueba_matriz  = [[0] * filas for _ in range(columnas)]
print(prueba_matriz)
prueba_matriz  = [[0] * 4]
print(prueba_matriz)
