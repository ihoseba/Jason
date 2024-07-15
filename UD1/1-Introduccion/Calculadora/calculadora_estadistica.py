""" Calculadora Estadistic """
import statistics
import os

# mean geometric_mean harmonia_mean
# median
# m ode

# statistics.mean()

def entr_datos():
    """ Funcion de entrada de una lista de datos
    devuelve una lista de datos """
    list_datos=[]
    entr_dato='f'
    n=0
    while entr_dato.lower() != '':
        n+=1
        entr_dato=input(f'{n}º dato: ')
        if entr_dato.lower() == '':
            break
        entr_num=int(entr_dato)
        list_datos.append(entr_num)
    return list_datos

#main
while True:
    # """ Bucle principal Main """
    os.system('cls')
    # introduzca set de datos en una lista
    # opciones o funcion a jectutar sobre ellos

    datos=entr_datos()
    print('los datos',datos)
    print('la media',statistics.mean(datos))
    operacion='ok'
    while operacion != 'x':
        print('operaciones disponibles (m)-Media, (g)-Media Geometrica, (h), (M), (d), (D), (s), (v), (x)-Salir')
        operacion=input('Que operacion quieres hacer? ')
        if operacion ==  'm':
            print('Media: ',statistics.mean(datos))
        elif operacion ==  'g':
            print('Media Geometrica: ',statistics.geometric_mean(datos))
        elif operacion ==  'h':
            print('Media harmonica: ',statistics.harmonic_mean(datos))
        elif operacion ==  'M':
            print('Mediana: ',statistics.median(datos))
        elif operacion ==  'd':
            print('Modo: ',statistics.mode(datos))
        elif operacion ==  'D':
            print('Desvio: ',statistics.pstdev(datos))
        elif operacion ==  's':
            print('Desvio Estandar: ',statistics.stdev(datos))
        elif operacion ==  'v':
            print('Varianza es: ',statistics.variance(datos))
        elif operacion == 'x':
            print (" Exit ")
            break
        else:
            continue
    if input('(Enter) para continuar con otro calculo, (x) para salir') == 'x':
        break
    continue
