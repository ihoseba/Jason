# Escribe un programa que cuente la frecuencia de cada carácter en una cadena.
def contar_caracteres(cadena):
    diccionario_frecuencia={}
    for car in cadena:
        if diccionario_frecuencia.get(car)==None:
            diccionario_frecuencia[car]=1
        else:
            diccionario_frecuencia[car]+=1
    return diccionario_frecuencia

# Prueba la función con una cadena. Debería devolver un diccionario parecido a este:
# {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}

cadena='pruebapru'
print(cadena)
print(contar_caracteres(cadena))

cadena='en un lugar de la mancha'
print(cadena)
print(contar_caracteres(cadena))
