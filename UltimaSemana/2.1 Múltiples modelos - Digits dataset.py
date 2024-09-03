#Funciones a emplear
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

models = [
    ('svc-rbf', SVC(kernel="rbf", random_state=42)),
    ('svc-linear', SVC(kernel="linear", random_state=42)),
    ('svc-poly', SVC(kernel="poly", random_state=42)),
    ('Dtree', DecisionTreeClassifier(random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LogisticRegression', LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)),
    ('KNN', KNeighborsClassifier()),
]

#Generamos el dataset y los suboconjuntos de entrenamiento y prueba
X,y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)

import numpy as np

new_map = []
for number in range(len(X)):
    number_base  = number * 8
    for n1 in range(8):
        for n2 in range(8):
            i = ((n1 -1) * 8) + n2
            val =X[number][i]
        
            new_i = number_base + n2
            if (len(new_map) -1 < new_i):
                new_map.append([])
    
            new_map[number_base + n1].append(val)

new_map = np.array(new_map)  # Convert the list of lists to a NumPy array

for (name, clf) in models:
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(name, accuracy_score(y_test, y_pred))

# Graficamos la matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['0','1','2','3','4','5','6','7','8','9'],
            yticklabels=['0','1','2','3','4','5','6','7','8','9'])
plt.xlabel('true label')
plt.ylabel('predicted label ')
plt.show()
