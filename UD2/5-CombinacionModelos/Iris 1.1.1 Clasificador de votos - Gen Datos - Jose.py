"""
 Clasificador de Votos
 Con Iris
 
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mglearn

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data[:,[1,2]]  # Tomar características (2 columnas)
y = iris.target

# Genera un plot
fig, ax = plt.subplots(figsize=(6,4))

print(fig, ax)


mglearn.discrete_scatter(X[:,0], X[:,1],y,ax=ax)
ax.set_title("Dataset Iris")
ax.xlabel="sepal length (cm)"
ax.ylabel="sepal width (cm)"

plt.show()

print(iris.data)
print(iris.feature_names)

