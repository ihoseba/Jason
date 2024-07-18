"""
Clasificador de Votos
con Iris

    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica
"""
#Funciones a emplear
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

#Clasificadores a emplear: regresión logística, RandomForest y SVC
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

# Cargar el conjunto de datos Iris
iris = load_iris()
X_prescaled = iris.data  # Tomar todas las características (4 columnas)
y = iris.target

#Escalar
scaler = StandardScaler()
X = scaler.fit_transform(X_prescaled)
# X = X_prescaled

#Generamos el dataset y los suboconjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

#Generamos el objeto VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
    ('lr', log_clf),
    ('rf', rnd_clf),
    ('svc',svm_clf
     )],
    voting='hard'
)

for clf in (log_clf, rnd_clf, svm_clf):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(f"{clf.__class__.__name__}: {accuracy_score(y_test, y_pred):.2f}")

voting_clf.fit(X_train,y_train)
y_pred = voting_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
