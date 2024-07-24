import mglearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#Obtenemos el conjunto de datos
iris=load_iris()
X=iris.data[:,2:]

#Creamos la clase KMeans y la instanciamos
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)

print("Los centroides están en:",kmeans.cluster_centers_)
X_nuevo=np.array([[2.5,1]])
print("El clúster asociado es: ",kmeans.predict(X_nuevo))
print("La distancia a los centroides es:",kmeans.transform(X_nuevo))

mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,markers="o")


mglearn.discrete_scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    [0,1, 2],
    markers="^",markeredgewidth=4
 )
