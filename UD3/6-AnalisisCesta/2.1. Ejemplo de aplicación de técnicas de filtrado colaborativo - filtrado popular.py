# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:52:12 2024

@author: Markel
"""

intereses_usuarios = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

from collections import Counter
popularidad_materia = Counter(interes
                              for intereses_usuarios in intereses_usuarios
                              for interes in intereses_usuarios)

#Necesitamos ahora las clases List y Tuple del método typing de la librería 
# estándar de Python, que nos permitirán convertir en una lista la dupla
# correspondiente a los intereses de una persona usuaria y el número máximo
# de recomendaciones que haremos (en este caso las fijaremos en 5)
from typing import List, Tuple
def nuevos_intereses_mas_populares(
    intereses_usuario: List[str],
    max_results: int = 5) -> List[Tuple[str, int]]:

    suggestions = [(interest, frequency)
                   for interest, frequency in popularidad_materia.most_common()
                   if interest not in intereses_usuario]

    return suggestions[:max_results]

i=0
print("Los intereses del usuario",i,"son:",intereses_usuarios[i])
print("A este usuario le recomendamos:", nuevos_intereses_mas_populares(intereses_usuarios[i]))