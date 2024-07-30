from sklearn.datasets import make_circles, make_blobs
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(20)

X_circles, y_circles = make_circles(n_samples=400, factor=0.5, noise=0.05)

centros_elipses = np.array([[4, 4], [6, 2]])
blob_std_elipses = np.array([0.5, 0.2])
X_elipses, y_elipses = make_blobs(n_samples=400, centers=centros_elipses, cluster_std=blob_std_elipses, random_state=20)

X_line = np.array([np.linspace(-3, 3, 400), 0.5 * np.linspace(-3, 3, 400)]).T

X_complex = np.vstack((X_circles, X_elipses, X_line))
y_complex = np.hstack((y_circles, y_elipses, np.full(400, 2)))

plt.figure(figsize=(6, 6))
plt.scatter(X_complex[:, 0], X_complex[:, 1], s=10)
plt.title("Dataset Complejo")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()
