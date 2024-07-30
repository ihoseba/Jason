import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
    
#Generación de la matriz
np.random.seed(0)
X = np.random.rand(10,10)*10
X_const=np.around(X[:])
#Primer biclúster
rows = np.array([0,1,2])[:, np.newaxis]
columns = np.array([0,1, 2, 3])
X_const[rows,columns] = [20,20,20,20]
#Segundo biclúster
rows1 = np.array([3,4,5])[:, np.newaxis]
columns1 = np.array([4, 5, 6])
X_const[rows1,columns1] = [30,30,30]
#Tercer biclúster
rows2 = np.array([6,7,8,9])[:, np.newaxis]
columns2 = np.array([7, 8, 9])
X_const[rows2,columns2] = [40,40,40]
print(X_const)
plt.matshow(X_const, cmap=plt.cm.Greens)
    
#Empleamos una variable auxiliar aleatoria, rng
rng = np.random.RandomState(0)
#Obtenemos permutaciones de las filas y las columnas
row_idx = rng.permutation(X_const.shape[0])
col_idx = rng.permutation(X_const.shape[1])
#Reconstruimos la matriz con la nueva distribución de filas y columnas
X_const = X_const[row_idx][:, col_idx]
print("")
print(X_const)
plt.matshow(X_const, cmap=plt.cm.Blues)
    
    
model = SpectralCoclustering(n_clusters=3, random_state=0)
model.fit(X_const)
fit_data = X_const[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]
print(fit_data)
plt.matshow(fit_data, cmap=plt.cm.Reds)
