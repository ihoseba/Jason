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

i=5
print("Los intereses del usuario",i,"son:",intereses_usuarios[i])
print("A este usuario le recomendamos:", nuevos_intereses_mas_populares(intereses_usuarios[i]))