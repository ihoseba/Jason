from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Iris
iris = load_iris()
X_prescaled = iris.data[:,[1,2]]  # Tomar características (2 columnas)
y = iris.target

#Escalar
scaler = StandardScaler()
X = scaler.fit_transform(X_prescaled)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

#Creamos el objeto BaggingClassifier
bag_clf = BaggingClassifier(SVC(random_state=42), n_estimators=500, max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

#Medimos la exactitud del ensamble
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Definición de la función
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
        plt.axis(axes)

#Entrenamos el modelo SVC
svm_clf = SVC(gamma="scale", probability = True, random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_svm))


fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(svm_clf, X, y)
plt.title("SVM", fontsize=14)
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X, y)
plt.title("SVM con Bagging", fontsize=14)
plt.ylabel("")