""" Funcion contacr caracteres """
# Escribe un programa que cuente la frecuencia de cada carácter en una CADENA.
# prueba github comit
def contar_caracteres(con_cadena):
    """ Contar Caracteres """
    diccionario_frecuencia={}
    for car in con_cadena:
        if diccionario_frecuencia.get(car) is None:
            diccionario_frecuencia[car]=1
        else:
            diccionario_frecuencia[car]+=1
    return diccionario_frecuencia

# Prueba la función con una CADENA. Debería devolver un diccionario parecido a este:
# {'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}

CADENA='pruebapru'
print(CADENA)
print(contar_caracteres(CADENA))

CADENA='en un lugar de la mancha'
print(CADENA)
print(contar_caracteres(CADENA))
