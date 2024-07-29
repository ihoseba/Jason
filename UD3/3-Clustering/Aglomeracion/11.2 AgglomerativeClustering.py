from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import mglearn
import numpy as np
centros_blob = np.array([[-12,-5],[-6,2],[3,0]])
blob_std = np.array([1, 2.5, 0.5])
X, y = make_blobs(n_samples=200, centers=centros_blob,
cluster_std=blob_std,random_state=20)
mglearn.discrete_scatter(X[:,0],X[:,1])
#Aplicamos agrupamiento por aglomeración
agg=AgglomerativeClustering(n_clusters=3)
agg.fit(X)
mglearn.discrete_scatter(X[:,0],X[:,1],agg.labels_)

agg=AgglomerativeClustering(n_clusters=3,linkage="single")