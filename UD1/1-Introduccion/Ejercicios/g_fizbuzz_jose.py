""" Modulo Fizz Buzz """
# Escribe un programa que imprima los números del 1 al 100, 
# pero para los múltiplos de 3 imprima "Fizz", 
# para los múltiplos de 5 imprima "Buzz", 
# y para los múltiplos de ambos imprima "FizzBuzz".

def es_multiplo(num,mul):
    """ Comprueba si es multiplo """
    if num % mul == 0:
        return True
    else:
        return False

def fizz_buzz(num):
    """ Comprueba Fizz Buzz """
    if es_multiplo(num,3) and es_multiplo(num,5):
        print("FizzBuzz") 
    elif es_multiplo(num,3):
        print("Fizz") 
    elif  es_multiplo(num,5):
        print("Buzz") 
    else:
        print(num)


for n in range(1,100+1):
    fizz_buzz(n)
