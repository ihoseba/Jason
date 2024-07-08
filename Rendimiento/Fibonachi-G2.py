"""Actividad práctica: código alternativo
Trata ahora de elaborar un código alternativo para la determinación de elementos de la sucesión de Fibonacci sin emplear una 
fórmula como la descrita en el enunciado del anterior caso práctico, pero mucho más eficiente que el basado en la recursividad.
Piensa en un bucle que vaya generando cada elemento hasta alcanzar aquel en el que estamos interesados. Comprueba cuánto tiempo
tarda ahora en calcular el elemento 40 de la sucesión. Una vez realizado, comprueba la solución"""

import time

""" Fibonacci """

def fibonac(n):
    if n < 2:
        return n
    resultado_previo=0
    resultado_ultimo=1
    for i in range(1,n):
        resultado=resultado_previo + resultado_ultimo
        resultado_previo = resultado_ultimo
        resultado_ultimo = resultado
    
    return resultado

mucho=False
n=0
resultado=0
while mucho is False and resultado<10**300:
    n += 1

    # Inicia el contador
    inicio = 1000*time.time()

    resultado = fibonac(n)

    # Detiene el contador e imprime el resultado
    fin = 1000*time.time()
    tiempo_transcurrido = fin - inicio

    if tiempo_transcurrido > 10:
        mucho=True


print(f"Iteracion: {n}")
print(f"El tiempo de ejecución es: {tiempo_transcurrido:.3f} mili segundos")
print(f"El término cuando n = {n} es:", resultado)
print(fin)
