"""
 Clasificador de Votos
 Con Iris
 
 
:Number of Instances: 150 (50 in each of three classes)
:Number of Attributes: 4 numeric, predictive attributes and the class
:Attribute Information:
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica

:Summary Statistics:
 
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mglearn

# sepal length in cm
# sepal width in cm
# petal length in cm
# petal width in cm
id_x=2
id_y=3

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data[:,[id_x,id_y]]  # Tomar características (2 columnas)
y = iris.target

# Genera un plot
fig, ax = plt.subplots(figsize=(6,4))

print(type(fig), type(ax))

mglearn.discrete_scatter(X[:,0], X[:,1],y,ax=ax)
ax.set_title("Dataset Iris")
print(iris.feature_names[id_x])
print(iris.feature_names[id_y])
plt.xlabel=iris.feature_names[id_x] # Esto no funciona ...
plt.ylabel=iris.feature_names[id_y] # Esto no funciona ...

plt.show()

