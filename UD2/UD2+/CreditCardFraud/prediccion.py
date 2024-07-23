"""
copilot en python recoge datos de un fichero denominado creditcard_reduced.csv y haz el codigo python necesario para lo siguiente
splitea lo datos
haz predicciones del resultado Class con varios modelos un modelo polynomico otro rbf utilizando las dos primeras columnas 
Posteriormente plotea el diagrama de fronteras para cada caso para los dos modelos 
y que sea rapidito

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Cargar datos desde creditcard_reduced.csv
data = pd.read_csv('creditcard_reduced+.csv')

# Dividir datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_prescaled = data.iloc[:, :2]  # Usamos las dos primeras columnas como características
y = data['Class']
#Escalar
scaler = StandardScaler()
X_array = scaler.fit_transform(X_prescaled)
X=pd.DataFrame(X_array,columns=["Columna1","Columna2"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo polinómico
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

model_poly2=SVC(kernel="poly", degree=3)
model_poly2.fit(X_train_poly, y_train)

# Modelo RBF
model_rbf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=1000)
model_rbf.fit(X_train, y_train)

model_rbf2=SVC(kernel="rbf")
model_rbf2.fit(X_train_poly, y_train)

# Ploteo de fronteras de decisión
print(X.min().Columna1, X.max().Columna1)
print(np.linspace(X.min().Columna1, X.max().Columna1))
print(np.linspace(X.min()["Columna2"], X.max()["Columna2"]))

xx, yy = np.meshgrid(np.linspace(X.min().Columna1, X.max().Columna1, 100),
                     np.linspace(X.min()["Columna2"], X.max()["Columna2"], 100))
Z_poly = model_poly2.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()]))
Z_rbf = model_rbf2.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_poly.reshape(xx.shape), cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.title("Frontera de decisión (Modelo polinómico)")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_rbf.reshape(xx.shape), cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.title("Frontera de decisión (Modelo RBF)")

plt.show()



y_pred=model_rbf2.predict(X_test)
print(f"Model rbf: {accuracy_score(y_test, y_pred):.2f}")
X_test_poly = poly.fit_transform(X_test)
y_pred=model_poly2.predict(X_test_poly)
print(f"Model poly: {accuracy_score(y_test, y_pred):.2f}")
