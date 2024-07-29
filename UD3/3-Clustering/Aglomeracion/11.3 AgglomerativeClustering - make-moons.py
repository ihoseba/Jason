from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
import mglearn

'''
El algoritmo de agrupamiento por aglomeración funciona muy bien en todas las situaciones
planteadas. Por desgracia, no proporciona resultados interesantes cuando el dataset tiene
formas complejas, como es el caso del conjunto moons, en el que emplearías el siguiente código
y obtendrías la salida gráfica que tienes tras él:
'''

X, y = make_moons(n_samples=300,noise=0.01,random_state=20)
mglearn.discrete_scatter(X[:,0],X[:,1])
agg=AgglomerativeClustering(n_clusters=2)
agg.fit(X)
mglearn.discrete_scatter(X[:,0],X[:,1],agg.labels_)