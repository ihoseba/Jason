from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward
import numpy as np
centros_blob = np.array([[-12,-5],[-6,2],[3,0]])
blob_std = np.array([1, 2.5, 0.5])
X, y = make_blobs(n_samples=15, centers=centros_blob,
cluster_std=blob_std,random_state=20)
#Obtenemos el array de enlaces entre instancias
array_enlaces=ward(X)
#Instanciamos el dendograma
dendrogram(array_enlaces)