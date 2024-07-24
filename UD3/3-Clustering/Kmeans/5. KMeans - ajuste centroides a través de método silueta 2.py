import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl

#Generamos el conjunto de datos
centros_blob = np.array([[1.5,2.4],[0.5,2.3],[-0.5,2],[-1,3],[-1.5,2.6]])
blob_std = np.array([0.3, 0.25, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=800, centers=centros_blob, cluster_std=blob_std,random_state=20)

#Creamos las clases KMeans para cada agrupación y las instanciamos
kmeans_por_k = [KMeans(n_clusters=k, random_state=20).fit(X)
                for k in range(1, 7)]
siluetas = [silhouette_score(X, model.labels_) for model in kmeans_por_k[1:]]

#Representación gráfica del coeficiente de silueta
plt.figure(figsize=(11, 9))
for k in (3, 4, 5, 6):
    # Para que la siguiente instrucción funcione necesitas que los valores de k sean correlativos ya que estás indicando las posiciones de cada gráfica
    plt.subplot(2, 2, k-2)
    y_pred = kmeans_por_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)
    padding = len(X) // 30
    pos = padding

    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()
        color = mpl.cm.Spectral(i / k)
        
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
        facecolor = color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding
        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        
        if k in (3, 5):
            plt.ylabel("Cluster")
        
        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Coeficiente silueta")
        else:
            plt.tick_params(labelbottom=False)
            plt.axvline(x=siluetas[k - 2], color="red", linestyle="--")
            plt.title("$k={}$".format(k), fontsize=16)