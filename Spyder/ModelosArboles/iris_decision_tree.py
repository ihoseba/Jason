# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:54:52 2024

@author: joseangelperez
"""
"""
Al igual que sucedía en los problemas de regresión, lo habitual en el contexto
de árboles es emplear, para el entrenamiento del algoritmo, un subconjunto de
los elementos a clasificar, de modo que los restantes se pueden utilizar para
las pruebas. En este caso, te animo a que utilices los procedimientos que ya
conoces para obtener los datos de entrenamiento del conjunto iris y que generes
el árbol sin fijar un nivel máximo de profundidad, aplicando como medida de
impureza el coeficiente de Gini.
Escribe el código necesario y obtén la gráfica de dicho árbol. 
¿Qué profundidad final se alcanza y cómo son las últimas hojas?
Responde a la cuestión expuesta y, posteriormente, comprueba tu respuesta.
"""

#Importamos el conjunto de datos y la función requerida para la elaboración del árbol
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from pydotplus import graph_from_dot_data

#Cargamos en X la longitud y anchura del pétalo, tal y como hemos hecho en secciones anteriores
#En y se vuelca la clasificación de cada lirio (0, 1, 2)
iris=load_iris()
X=iris.data[:,2:]
y=iris.target

#Creamos un objeto arbol_clf con todos los atributos asociados a la construcción del DecisionTreeClassifier
arbol_clf=DecisionTreeClassifier(max_depth=2)
arbol_clf.fit(X,y)

export_graphviz(arbol_clf,out_file="arbol_iris.dot",feature_names=iris.feature_names[2:],
class_names=iris.target_names,rounded=True,filled=True)

arbol_data = export_graphviz(arbol_clf,out_file=None,feature_names=iris.feature_names[2:], class_names=iris.target_names,rounded=True,filled=True)

grafica = graph_from_dot_data(arbol_data)

grafica.write_png("arbol.png")


# para el entrenamiento del algoritmo, un subconjunto de los elementos a clasificar
# restantes se pueden utilizar para las pruebas.
# En este caso, te animo a que utilices los procedimientos que ya conoces para
# obtener los datos de entrenamiento del conjunto iris y
# que generes el árbol sin fijar un nivel máximo de profundidad, aplicando Gini.
# Escribe el código necesario y obtén la gráfica de dicho árbol. 
# ¿Qué profundidad final se alcanza y cómo son las últimas hojas?
# Responde a la cuestión expuesta y, posteriormente, comprueba tu respuesta.



