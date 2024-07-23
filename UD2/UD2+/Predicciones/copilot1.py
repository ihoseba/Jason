"""
Copilot genera un codigo python que recoja informacion del usuario solicitando en un menu
 dos funciones entre lineal, seno, hiperbolica, fibonacci y espiral
 y muestre en una imagen la linea resultante de la primera en el eje x y la segunda en el eje y
 que muestre en la leyenda los datos de entrada, así como las etiquetas de ejex x e y
 Tambien que plotee un set de 50 puntos random que los vizualice como triangulo o cuadrado
 en el gráfico en funcion de si esta a un lado o al otr o de ladicha linea de funciones


"""
import numpy as np
import matplotlib.pyplot as plt
import random

# Funciones
def lineal(x):
    return 2 * x + 3

def seno(x):
    return np.sin(x)

def hiperbolica(x):
    return np.cosh(x)

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def espiral(t):
    a = 0.1
    b = 0.2
    return a * np.exp(b * t) * np.cos(t), a * np.exp(b * t) * np.sin(t)

# Generar datos
x_vals = np.linspace(-10, 10, 1000)
y_lineal = lineal(x_vals)
y_seno = seno(x_vals)
y_hiperbolica = hiperbolica(x_vals)
y_fibonacci = [fibonacci(i) for i in range(10)]
t_espiral = np.linspace(0, 10, 1000)
x_espiral, y_espiral = espiral(t_espiral)

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_lineal, label="Lineal: 2x + 3")
plt.plot(x_vals, y_seno, label="Seno: sin(x)")
plt.plot(x_vals, y_hiperbolica, label="Hiperbólica: cosh(x)")
plt.scatter(x_espiral, y_espiral, c="red", marker="o", label="Espiral")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Gráfico de funciones")
plt.legend()

# Mostrar puntos aleatorios como triángulos o cuadrados
for _ in range(50):
    x_random = random.uniform(-10, 10)
    y_random = random.uniform(-10, 10)
    shape = "s" if x_random + y_random > 0 else "^"
    plt.scatter(x_random, y_random, c="blue", marker=shape)

plt.grid()
plt.show()
