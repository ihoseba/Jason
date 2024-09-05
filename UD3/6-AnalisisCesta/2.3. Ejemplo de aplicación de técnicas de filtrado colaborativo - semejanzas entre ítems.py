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
from collections import defaultdict
popularidad_materia = Counter(interes for intereses_usuarios in intereses_usuarios for interes in intereses_usuarios)

from numpy import dot
from numpy.linalg import norm
def similitud_coseno(v1, v2):
    return dot(v1, v2)/(norm(v1)*norm(v2))

intereses_unicos=sorted({interes for intereses_usuarios in intereses_usuarios for interes in intereses_usuarios})

def gen_vector_interes_usuario(intereses_usuarios):
    return [1 if interes in intereses_usuarios else 0 for interes in intereses_unicos]

vector_interes_usuario=[gen_vector_interes_usuario(intereses_usuarios)
for intereses_usuarios in intereses_usuarios]

# 1: Obtener una matriz con los intereses de cada individuo
matriz_intereses_usuarios = [[vector_interes_usuario[j]
                              for vector_interes_usuario in vector_interes_usuario]
                        for j, _ in enumerate(intereses_unicos)]
print(matriz_intereses_usuarios[:1])


# 2: Aplicar similitud de coseno
semejanzas_entre_tematicas = [[
    similitud_coseno(vector_interes_i, vector_interes_j)
    for vector_interes_j in matriz_intereses_usuarios]
    for vector_interes_i in matriz_intereses_usuarios]


def intereses_mas_similares_a(interes_id: int):
    semejanzas = semejanzas_entre_tematicas[interes_id]
    pares = [
        (intereses_unicos[otro_interes_id], semejanza)
        for otro_interes_id, semejanza in enumerate(semejanzas)
        if interes_id != otro_interes_id and semejanza > 0]

    return sorted(pares,
                  key=lambda pair: pair[-1],
                  reverse=True)

print("intereses_mas_similares_a(0)")
print(intereses_mas_similares_a(0))


# 3: Establecer recomendaciones
def sugerencias_basadas_item(usuario_id: int,
    incluir_intereses_actuales: bool = False):
    # Vamos agregando los intereses semejantes
    sugerencias = defaultdict(float)
    vector_intereses_usuario = [gen_vector_interes_usuario(intereses_usuario)
    for intereses_usuario in intereses_usuarios]
    vector_interes_usuario = vector_intereses_usuario[usuario_id]
    for interes_id, esta_interesado in enumerate(vector_interes_usuario):
        if esta_interesado == 1:
            intereses_similares = intereses_mas_similares_a(interes_id)
            for interes, semejanza in intereses_similares:
                sugerencias[interes] += semejanza
    # Ordenamos por pesos
    sugerencias = sorted(sugerencias.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)
    #Descartamos intereses que ya contemplaba la persona
    if incluir_intereses_actuales:
        return sugerencias
    else:
        return [(sugerencia, weight)
                for sugerencia, weight in sugerencias
                    if sugerencia not in intereses_usuarios[usuario_id]]

print(sugerencias_basadas_item(0))