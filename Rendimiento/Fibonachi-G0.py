import time

def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
   
n = int(input("dime el el nesimo fibonacci que deseas "))

# Inicia el contador
inicio = time.time()

resultado = fibonacci(n)
print(f"El término cuando n = {n} es:", resultado)

# Detiene el contador e imprime el resultado
fin = time.time()
tiempo_transcurrido = fin - inicio
print(f"El tiempo de ejecución es: {tiempo_transcurrido:.6f} segundos")