import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Iris
iris = load_iris()
X_prescaled = iris.data  # Tomar todas las características (4 columnas)
y = iris.target

#Escalar
scaler = StandardScaler()
X = scaler.fit_transform(X_prescaled)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

#Creamos el objeto BagClassifier
bag_clf=BaggingClassifier(
    DecisionTreeClassifier(
        splitter="random",
        max_leaf_nodes=16,
        random_state=42
    ),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=True,
    random_state=42
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("Accuracy Bagging: ",accuracy_score(y_test, y_pred))

#Creamos el objeto Random Forest
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

#Comparamos los resultados de Bagging y Random Forest
print("Accuracy Random Forest: ",accuracy_score(y_test, y_pred_rf))

#Medimos la importancia de cada característica
print("Importancia de Cada Caracteristica: ",rnd_clf.feature_importances_)