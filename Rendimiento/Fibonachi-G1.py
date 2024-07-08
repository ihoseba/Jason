import time

""" Fibonacci """
from math import sqrt

def fib(n):
    if n < 2:
        return n
    else:
        phi = (1 + sqrt(5)) / 2
        j = ((phi**n - (1 - phi)**n) / sqrt(5))
        return round(j)

# Ejemplo
n=int(input("Cual es n-esimo? "))

# Inicia el contador
inicio = time.time()

resultado = fib(n)
print(f"El término cuando n = {n} es: {resultado}")

# Detiene el contador e imprime el resultado
fin = time.time()
tiempo_transcurrido = fin - inicio
print(f"El tiempo de ejecución es: {tiempo_transcurrido:.6f} segundos")