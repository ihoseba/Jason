import time

def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
   
mucho=False
n=0
resultado=0
while mucho is False and resultado<1000:
    n += 1

    # Inicia el contador
    inicio = time.time()

    resultado = fibonacci(n)
    
    # Detiene el contador e imprime el resultado
    fin = time.time()
    tiempo_transcurrido = fin - inicio

    if tiempo_transcurrido > 10:
        mucho=True


print(f"Iteracion: {n}")
print(f"El tiempo de ejecución es: {tiempo_transcurrido:.1f} segundos")
print(f"El término cuando n = {n} es:", resultado)
