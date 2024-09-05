# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:16:10 2024

@author: joseangelperez
"""


intereses_usuarios =[
    ["Canta 2", "Encanto", "Diario de Greg", "Frozen 1"],
    ["Toy Story 4", "Los Minions", "Frozen 1"],
    ["Encanto", "Clifford", "Gru 2", "Gru 3", "Canta 2"],
    ["Hotel Transilvania", "Frozen 1", "Frozen 2"],
    ["Ron da error", "Gru 1", "Gru 2", "Los Minions", "Canta 2"],
    ["Encanto", "Diario de Greg", "Patrulla Canina", "Space Jam"],
    ["Gru 3", "Hotel Transilvania", "Los Minions"],
    ["Toy Story 4", "Clifford", "Gru 1", "Canta 2"],
    ["Hotel Transilvania", "Los Minions", "Gru 1", "Diario de Greg"],
    ["Toy Story 4", "Ron da error", "Space Jam", "Clifford", "Encanto"]
]
que_usuario=4

print("-------------------------------------------")
print("Basado en Popular")
print("-------------------------------------------")

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

i=que_usuario
print("Los intereses del usuario",i,"son:",intereses_usuarios[i])
print("A este usuario le recomendamos:", nuevos_intereses_mas_populares(intereses_usuarios[i]))

print("-------------------------------------------")
print("Basado en Usuarios")
print("-------------------------------------------")

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

vector_interes_usuario=[gen_vector_interes_usuario(intereses_usuario)
                        for intereses_usuario in intereses_usuarios]

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

usuarios_mas_similares_a_cero = usuarios_mas_similares_a(que_usuario)
print("usuarios_mas_similares_a_cero ",usuarios_mas_similares_a_cero)


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
            
print("sugerencias_basadas_usuarios", sugerencias_basadas_usuarios(que_usuario))

print("-------------------------------------------")
print("Basado en Items")
print("-------------------------------------------")

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

print("intereses_mas_similares_a(",que_usuario,")")
print(intereses_mas_similares_a(que_usuario))


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

print("sugerencias_basadas_item ",que_usuario)
print(sugerencias_basadas_item(que_usuario))
