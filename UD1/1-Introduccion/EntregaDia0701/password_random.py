""" generador de contraseñas aleatorias de n caracteres ascii caracteres + digitos """
import os
import string
import random
caracteres = string.ascii_letters + string.digits + "+-*/¡?)()/&%$·\"ºª"
os.system('cls')
while True:
    print("Poblacion de caracteres para esta funcion")
    print(caracteres)
    print('Generador de PW')
    num=input('(X para salir) Cuantos caracteres quieres generar?:')
    print(num,string.digits)
    if num.isdigit():
        num=int(num)
        pass_word=''
        for i in range(0,num):
            pass_word = pass_word + random.choice(caracteres)
        print(pass_word)
    elif num.lower()=='x':
        break
    else:
        print('ha de ser numerico')
        continue

