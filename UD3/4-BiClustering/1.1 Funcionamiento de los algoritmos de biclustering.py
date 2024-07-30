import numpy as np
#Inicializamos la semilla para que siempre que ejecutemos el código obtengamos la misma matriz
np.random.seed(0)
#La función rand genera un número aleatorio entre 0 y 1.
#En nuestro caso estamos interesados en obtener números entre 0 y 10, por lo que bastará con multiplicar cada valor por 10
#Indicamos que el array esté formando por 10 filas y 10 columnas
X = np.random.rand(10,10)*10
#Redondeamos los valores obtenidos para que simulemos que trabajamos con enteros. Empleamos la función around, que ya conoces
#Denotaremos como X_aleat a la matriz completamente aleatoria. La clonamos a partir de X para separar ambas matrices
#Recuerda que si hubieras hecho directamente X_aleat = X los cambios en X_aleat también los harías en X
X_aleat=np.around(X[:])
#Imprimimos la matriz
print(X_aleat)

#Indicamos las filas y columnas en las que vamos a forzar los nuevos valores
#Guardamos en un array las filas y columnas de las que se trata. Recuerda que la primera fila es la cero y la columna segunda se numera como la 1
#Con newaxis transformamos rows de un vector columna a un vector fila
rows = np.array([0,1,2,3])[:, np.newaxis]
columns = np.array([1, 2, 3])
#De nuevo clonamos la matriz original, sin redondear
X_const=np.around(X[:])
X_const[rows,columns] = [2,2,2]
print((X_const))

#Indicamos los nuevos valores de filas y columnas a modificar y clonamos la matriz original
rows1= np.array([6, 7, 8])[:, np.newaxis]
columns1 = np.array([5, 6, 7])
X_sum=np.around(X[:])
#Indicamos que los nuevos valores estén entre 1 y 10 y que vayan aumentando de 1 en 1
X_sum[rows1,columns1] = np.arange(1,10,1).reshape(3, 3)
print(X_sum)