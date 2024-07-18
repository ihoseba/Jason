#Funciones a emplear
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Clasificadores a emplear: regresión logística, RandomForest y SVC
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

#Generamos el dataset y los suboconjuntos de entrenamiento y prueba
X,y=make_moons(n_samples=500,noise=0.3,random_state=20)
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
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

voting_clf.fit(X_train,y_train)
y_pred = voting_clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))