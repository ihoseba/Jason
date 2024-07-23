"""
Actividad práctica: Titanic
Utiliza el dataset titanic disponible en kaggle. Clasifica el dataset utilizando:
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
La columna objetivo será: “survival”
Determina la exactitud alcanzada en cada caso y comenta los resultados.
Representa gráficamente las fronteras de decisión para cada método)

Copilot. Utiliza el dataset titanic disponible en kaggle.
Clasifica el dataset utilizando dos columnas de datos mas correladas
con el objetivo "survival":
    • SVM con kernel lineal, RBF y polinomial.
    • Un clasificador K-Nearest Neighbors (KNN).
    • Un árbol de decisión.
La columna objetivo será: “survival”
Representa gráficamente las fronteras de decisión para cada método y
Determina la exactitud alcanzada en cada caso. 

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from mlxtend.plotting import plot_decision_regions


file_path_train = 'train.csv'
titanic_train = pd.read_csv(file_path_train,usecols=['Survived','Pclass', 'Fare'])
file_path_test = 'test.csv'
titanic_test = pd.read_csv(file_path_test,usecols=['Pclass', 'Fare'])

# 'Pclass', 'Fare'
X=titanic_train.iloc[:,[1,2]]
y=titanic_train.iloc[:,[0]]
X_test_fin=titanic_test.iloc[:,[0,1]]
print(X)


"""
# Cargar el conjunto de datos Titanic
titanic_data = fetch_openml(name="titanic", version=1, as_frame=True)
X, y = titanic_data.data, titanic_data.target


# Seleccionar las columnas más correlacionadas con "survival"
selected_columns = ["Pclass", "Fare"]
X_subset = X[selected_columns]

"""

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = np.ravel(y_train)

# 1. SVM con kernel lineal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
print(f"Exactitud SVM (kernel lineal): {accuracy_svm_linear:.2f}")

# 2. SVM con kernel RBF
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
print(f"Exactitud SVM (kernel RBF): {accuracy_svm_rbf:.2f}")

# 3. Clasificador K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Exactitud KNN: {accuracy_knn:.2f}")

# 4. Árbol de decisión
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Exactitud Árbol de decisión: {accuracy_tree:.2f}")

# Representar gráficamente las fronteras de decisión (por ejemplo, SVM con kernel lineal)
# Puedes adaptar esto para los otros métodos también

# Ploteo de Pclass vs Fare
plt.figure(figsize=(10, 6))
plt.scatter(X_train["Pclass"], X_train["Fare"],c=y_train, cmap='viridis', edgecolor='k')
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.title("Clasificación de pasajeros del Titanic")
plt.show()

# Graficar las fronteras de decisión
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, clf=svm_linear, legend=2)
plt.xlabel("Pclass")
plt.ylabel("Fare")
# plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
plt.show()

# Graficar las fronteras de decisión
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, clf=svm_rbf, legend=2)
plt.xlabel("Pclass")
plt.ylabel("Fare")
# plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
plt.show()

# Graficar las fronteras de decisión
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, clf=knn, legend=2)
plt.xlabel("Pclass")
plt.ylabel("Fare")
# plt.title(f"Frontera de decisión ({name}) - Accuracy = {accuracy:.2f}")
plt.show()
