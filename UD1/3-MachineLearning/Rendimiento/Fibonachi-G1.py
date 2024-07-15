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

mucho=False
n=0
resultado=00
while mucho is False and resultado<10**300:
    n += 1

    # Inicia el contador
    inicio = time.time()

    resultado = fib(n)

    # Detiene el contador e imprime el resultado
    fin = time.time()
    tiempo_transcurrido = fin - inicio

    if tiempo_transcurrido > 10:
        mucho=True


print(f"Iteracion: {n}")
print(f"El tiempo de ejecución es: {tiempo_transcurrido:.1f} segundos")
print(f"El término cuando n = {n} es:", resultado)
