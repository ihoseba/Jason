""" Plotear"""
import os
import math
import matplotlib.pyplot as plt

# Main
os.system('cls')
# Introducir datos
in_datos=input('introducir datos separados por comas: ')
print(in_datos)
lista=in_datos.split(',')
print(lista)
lista=[int(i) for i in lista]
print(lista)
plt.plot(lista)
plt.show()
plt.savefig('lista.png')

# Convertir datos
#try, except

# Plotear datos

datos=[0,0]
for x in range(0,360):
    datos.append(math.sin(math.pi*(x/100)))

plt.plot(datos)
plt.show()
plt.savefig('datos.png')
