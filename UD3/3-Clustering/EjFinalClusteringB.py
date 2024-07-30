from sklearn.datasets import make_circles, make_blobs
from sklearn.datasets import make_circles, make_blobs
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(20)

X_circles, y_circles = make_circles(n_samples=400, factor=0.5, noise=0.05)

centros_elipses = np.array([[4, 4], [6, 2]])
blob_std_elipses = np.array([0.5, 0.2])
X_elipses, y_elipses = make_blobs(n_samples=400, centers=centros_elipses,
                                  cluster_std=blob_std_elipses, random_state=20)

X_line = np.array([np.linspace(-3, 3, 400), 0.5 * np.linspace(-3, 3, 400)]).T

# X_complex = np.vstack((X_circles, X_elipses, X_line))
# y_complex = np.hstack((y_circles, y_elipses, np.full(400, 2)))
X_complex = np.vstack((X_circles, X_line))
y_complex = np.hstack((y_circles, np.full(400, 2)))

# Plotear dataset con discriminacion
from seaborn import scatterplot
plt.figure(figsize=(10, 7))
scatterplot(x=X_complex[:, 0], y=X_complex[:, 1],c=y_complex)
plt.title(f"Datos Reales con {len(np.unique(y_complex))} clusters")
plt.show()

# Primero normalizar
from sklearn.preprocessing import StandardScaler
# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_complex)

# Plotear Normalizado
# Plotear Dataset
plt.figure(figsize=(6, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=10)
plt.title("Dataset Complejo Normalizado")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()
    
# Definir diferentes algoritmos
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering

algorithms = [
    DBSCAN(eps=.30, min_samples=7),
    KMeans(n_clusters=3,init='random',algorithm='elkan'),
    AgglomerativeClustering(n_clusters=3,linkage='ward'),
]

# Plotear diferentes algoritmos
import mglearn

fig, axes = plt.subplots(2, 2, figsize=(30, 10))
axes_flat  = axes.flatten()

for alg, ax in zip(algorithms,axes_flat):
    alg.fit(X_scaled)
    ya=alg.labels_
    mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1], y=ya, ax=ax)
    ax.set_title(f"{alg.__class__.__name__} - {len(np.unique(ya))} clusters")

# y los datos reales
ax=axes_flat[3]
mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1], y=y_complex, ax=ax)
ax.set_title("Datos Reales con 5 clusters")

# Hacemos primero una division de datos en tres grupos
