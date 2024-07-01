""" Calculadora en Python"""
# entrada teclado comando por pulsacion unica
# # Menu de operaciones +,-,*,/
# dos numeros de entrada
# vuelta al inicio con opcion de salir
# Añadir comprobacion si resultado es par o impar, mayor o menor de cero

# funciones

# main
# import msvcrt
import os
import fractions

ERROR=666
# ceil, floor, sqrt, exp, cos, pi
def menu():
    """" Menu de la calculadora """
    os.system('cls')
    OPCIONES=['+','-','*','/','4','c','C','f','F','s','S','e','E','o','O','p','P']
    print("Opciones (+),(-),(*),(/),(c)eil,(f)loor,(s)qrt,(e)sp,c(o)s,(p)i (Salir)")
    opcion="ERROR" # hasta que sepa hacerlo mejor
    while opcion not in OPCIONES:
        opcion=input("Operacion?: ")
        print(opcion)
#        opcion=int(opcion)
        if opcion not in OPCIONES:
            print('Ha de ser:')
            print("Opciones (+),(-),(*),(/),(c)eil,(f)loor,(s)qrt,(e)sp,c(o)s,(p)i (Salir)")
            opcion=ERROR
    return opcion

def numero():
    """ Solicita un numero """
    # while num is not float or num is not int:
    # num=float(input(" numerico eh!: "))
    # return num
    num=input(" numerico eh!: ")
    if '.' in num:
        num = float(num)
    else:
        num = int(num)
    return num

def operandos():
    """ Solicita dos operandos """
    print(" Ahora dame primer operando")
    a=numero()
    print(type(a))
    print(" Y el segundo")
    b=numero()
    return a,b

def frac_operandos():
    """ Solicita dos Fracciones  """
    print(" Ahora dame primer operando fraccion")
    a=in_fraccion()
    print(type(a))
    print(" Y el segundo")
    b=in_fraccion()
    return a,b

def suma(a,b):
    print(type(a),type(b)) # +++
    return(a+b)
def resta(a,b):
    return(a-b)
def multiplicacion(a,b):
    return(a*b)
def division(a,b):
    res=a/b # Por defecto res será float
    if res==round(res,0):
        res=int(res) # si no contiene decimales lo convertimos a int
    return(res)
def op_ceil(n):
    """ Calcula Ceil """
    return(n)
def op_floor(n):
    """ Calcula Floor """
    return(n)
def op_sqrt(n):
    """ Calcula Raiz """
    return(n)
def op_exp(n):
    """ Calcula Exponencial """
    return(n)
def op_cos(n):
    """ Calcula Coseno """
    return(n)
def op_pi(n):
    """ Devuelve Pi """
    return(n)

def evaluacion(valor):
    print('este valor es')
    if valor > 0:
        print(" Positivo ")
    elif valor<0:
        print(" Negativo ")
    else:
        print('es cero')

    if valor==round(valor,0):
        print('si entero')
        if valor%2 == 0:
            print('par')
        else:
            print('impar')
    else:
        print('si real')

    if isinstance(valor,int):
        print('entero')
    elif isinstance(valor,float):
        print('real')
    else:
        print('no se')

def in_fraccion():
    """ Solicita Fraccion """
    in_frac=input("Dame fraccion en formato 12/45:")
    chequear=['1','2','3','4','5','6','7','8','9','0','/']
    for val in in_frac:
        if val in chequear:
            None
        else:
            print('Errorr')
    resultado = fractions.Fraction(in_frac)
    return(resultado)

opcion=ERROR # no se hacerlo mejor por ahora :-)
while opcion != '0':
    # os.system('cls')
    # fraccion1 = in_fraccion()
    # print(fraccion1)
    # fraccion2 = in_fraccion()
    # print(fraccion2)

    # print(fraccion1+fraccion2)

    # resdo=fraccion1+fraccion2
    
    # print('resultado:',resdo)
    opcion=menu()
    if opcion=='0':
        print("Saliendo...")
        break
    if opcion=='+' or opcion=='1':
        a,b=frac_operandos()
        resultado=suma(a,b)
        print(type(a),type(b),type(resultado))
        print(f'La Suma de {a} + {b} es {resultado}')
    elif opcion=='-' or opcion=='2':
        a,b=frac_operandos()
        resultado=resta(a,b)
        print(f'La Resta de {a} - {b} es {resultado}')
    elif opcion=='3*' or opcion=='3':
        a,b=frac_operandos()
        resultado=multiplicacion(a,b)
        print(f'La Multiplicacion de {a} x {b} es {resultado}')
    elif opcion=='/' or opcion=='4':
        a,b=frac_operandos()
        resultado=division(a,b)
        print(f'La Division de {a} : {b} es {resultado}')
    elif opcion=='c':
        a,b=operandos()
        resultado=op_ceil(a,b)
        print(f'La Division de {a} : {b} es {resultado}')
    else:
        print("ERROR")
    evaluacion(resultado)
    input("Enter, continuar...")


