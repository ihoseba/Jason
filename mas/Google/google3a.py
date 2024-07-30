# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:48:38 2024

@author: joseangelperez

copilot escribe codigo python para lo siguiente coge un csv llamado GOOG.csv
carga los datos Date,Open,High,Low,Close,Adj Close,Volume
genera un nuevo set de datos que contenga para cada Date Close como resultado y
Open, High,Low,Close,Adj Close,Volume, de ese dia y de los 7 dias anteriores
Plotea diariamente Open, Close en eje y sobre Date diario en eje x
Aplica un algoritmo de regresion de sklearn y haz una estimacion para los siguientes
7 dias, ploteados en otro color en la misma grafica
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV
df = pd.read_csv("GOOG.csv", parse_dates=["Date"], dayfirst=True)

# Crear una nueva columna con la media de las variables Close, Open, High, Low, Adj Close y Volume de los 7 días anteriores
for i in range(1, 8):
    df[f"Close_{i}"] = df["Close"].shift(i)
    df[f"Open_{i}"] = df["Open"].shift(i)

# Filtrar las filas con NaN (debido al desplazamiento)
df = df.dropna()

# Crear un conjunto de datos con las columnas Close y Open de los 7 días anteriores
X = df[["Close_" + str(i) for i in range(1, 8)] + ["Open_" + str(i) for i in range(1, 8)]]
y = df["Close"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir los valores de los próximos 7 días
next_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
predicted_close = model.predict(next_days)

# Crear una figura
plt.figure(figsize=(10, 6))

# Gráfico de líneas para Open y Close
plt.plot(df["Date"], df["Open"], label="Open", linestyle="-", color="blue")
plt.plot(df["Date"], df["Close"], label="Close", linestyle="-", color="green")

# Gráfico de líneas para las estimaciones de los próximos 7 días
plt.plot(pd.date_range(start=df["Date"].iloc[-1], periods=7, closed="right"), predicted_close, label="Predicted Close", linestyle="--", color="red")

# Configurar leyendas y etiquetas
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Prices and Predictions")

# Mostrar el gráfico
plt.tight_layout()
plt.show()

