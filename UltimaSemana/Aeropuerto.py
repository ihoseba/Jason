# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:47:04 2024

@author: joseangelperez

codigo python para dataset de sklearn que prediga comportamiento compradores en aeropuerto

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos (ficticio)
# Supongamos que tienes un DataFrame llamado 'data' con las características y la etiqueta 'target'
# data = pd.read_csv('ruta_a_tu_dataset.csv')

# Para este ejemplo, crearemos un conjunto de datos ficticio
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'feature3': np.random.rand(1000),
    'feature4': np.random.rand(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Separar características y etiquetas
X = data.drop('target', axis=1)
y = data['target']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir los modelos
modelos = {
    "SVC (linear)": SVC(kernel='linear'),
    "SVC (rbf)": SVC(kernel='rbf'),
    "SVC (poly)": SVC(kernel='poly'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier()
}

# Evaluar el rendimiento de cada modelo y plotear la matriz de confusión
resultados = {}
plt.figure(figsize=(20, 12))

for i, (nombre, modelo) in enumerate(modelos.items(), 1):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    resultados[nombre] = score
    
    # Plotear la matriz de confusión
    plt.subplot(3, 3, i)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{nombre} (Accuracy: {score:.4f})")
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()

# Mostrar los resultados
for nombre, score in resultados.items():
    print(f"{nombre}: {score:.4f}")

# Encontrar el mejor modelo
mejor_modelo = max(resultados, key=resultados.get)
print(f"\nEl mejor modelo es: {mejor_modelo} con una precisión de {resultados[mejor_modelo]:.4f}")

# Ploteo de aciertos, errores y falsos positivos
plt.figure(figsize=(20, 12))

for i, (nombre, modelo) in enumerate(modelos.items(), 1):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular aciertos, errores y falsos positivos
    aciertos = np.diag(cm)
    errores = cm.sum(axis=1) - aciertos
    falsos_positivos = cm.sum(axis=0) - aciertos
    
    plt.subplot(3, 3, i)
    plt.bar(['Aciertos', 'Errores', 'Falsos Positivos'], [aciertos.sum(), errores.sum(), falsos_positivos.sum()], color=['green', 'red', 'orange'])
    plt.title(f"{nombre}")
    plt.ylabel('Cantidad')

plt.tight_layout()
plt.show()