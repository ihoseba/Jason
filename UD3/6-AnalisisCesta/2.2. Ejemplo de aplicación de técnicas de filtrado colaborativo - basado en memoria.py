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
popularidad_materia = Counter(interes for intereses_usuarios in intereses_usuarios
                              for interes in intereses_usuarios)

# 1: Aplicación de índice de similitud
from numpy import dot
from numpy.linalg import norm
def similitud_coseno(v1, v2):
    return dot(v1, v2)/(norm(v1)*norm(v2))

v1=(1,0)
v2=(2,0)
v3=(0,1)
v4=(-1,0)

print("similitud_coseno(v1,v2)", similitud_coseno(v1,v2))
print(similitud_coseno(v1,v3))
print(similitud_coseno(v1,v4))

# 2. Obtener vectores de similitud
intereses_unicos=sorted({interes for intereses_usuarios in intereses_usuarios
                         for interes in intereses_usuarios})
print(len(intereses_unicos))
print("intereses_unicos")
print(intereses_unicos)

def gen_vector_interes_usuario(intereses_usuario):
    return [1 if interes in intereses_usuario else 0 for interes in intereses_unicos]

vector_interes_usuario=[gen_vector_interes_usuario(intereses_usuario) for intereses_usuario in intereses_usuarios]

print("vector_interes_usuario")
print(vector_interes_usuario)


# 3: Medir la similitud entre las preferencias de distintos individuos

semejanzas_usuarios = [[similitud_coseno(vector_interes_i, vector_interes_j)
for vector_interes_j in vector_interes_usuario]
for vector_interes_i in vector_interes_usuario]


# 4: Crear una función que identifique las personas más cercanas, en gustos, a una dada
def usuarios_mas_similares_a(usuario_id: int):
    pares = [(otro_usuario_id, similitud)
             for otro_usuario_id, similitud in
             enumerate(semejanzas_usuarios[usuario_id])
             if usuario_id != otro_usuario_id and similitud > 0]
    
    #Con el siguiente código devolvemos el listado de los usuarios con gustos más próximos, de mayor a menor cercanía
    return sorted(pares,
                  key=lambda pair: pair[-1],
                  reverse=True)

usuarios_mas_similares_a_cero = usuarios_mas_similares_a(0)
print(usuarios_mas_similares_a_cero)


# 5: Definir una función de sugerencias basadas en las preferencias de cada persona
#Necesitamos importar varias funciones y clases
#Con defaultdict podremos transformar una lista en un diccionario
from collections import defaultdict
from typing import Dict, List, Tuple
#Definición de la función
def sugerencias_basadas_usuarios(usuario_id: int,
    incluir_intereses_actuales: bool = False):

    # Vamos agregando las semejanzas
    sugerencias: Dict[str, float] = defaultdict(float)

    for otro_usuario_id, similitud in usuarios_mas_similares_a(usuario_id):
        for interes in intereses_usuarios[otro_usuario_id]:
            sugerencias[interes] += similitud

            # Ordenamos las semejanzas
            sugerencias = sorted(sugerencias.items(),
                key=lambda pair: pair[-1],
                reverse=True)

            # Excluimos en su caso intereses ya declarados por parte de la persona en cuestión
            if incluir_intereses_actuales:
                return sugerencias
            else:
                return [(sugerencias, weight)
                     for sugerencias, weight in sugerencias
                     if sugerencias not in intereses_usuarios[usuario_id]]
            
print("sugerencias_basadas_usuarios", sugerencias_basadas_usuarios(0))