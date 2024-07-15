# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:18:32 2024

@author: joseangelperez
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from pydotplus import graph_from_dot_data

iris=load_iris()

X=iris.data[:,2:]
y=iris.target

arbol_clf=DecisionTreeClassifier(max_depth=2)
arbol_clf.fit(X,y)

out_graph=export_graphviz(
    arbol_clf,
    out_file=None,
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True)

graph=graph_from_dot_data(out_graph)
graph.write_png('arbol.png')

