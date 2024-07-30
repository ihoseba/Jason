# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:48:38 2024

@author: joseangelperez

copilot escribe codigo python para lo siguiente coge un csv llamado GOOG.csv
carga los datos Date,Open,High,Low,Close,Adj Close,Volume
genera un nuevo set de datos que contenga para cada Date, Close como resultado y
Open,High,Low,Close,Adj Close,Volume, de ese dia (Date) y tambien de los 7 dias
anteriores en mas columnas
Aplica un algoritmo de regresion de sklearn para cada columna 
Date,Open,High,Low,Close,Adj Close,Volume
Seguidamente haz la estimacion para cada columna Date,Open,High,Low,Close,Adj Close,Volume 
para la ultima fecha Date en el dataset, imprime  los valores estimados
Date,Open,High,Low,Close,Adj Close,Volume 

"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

# Crear un modelo de regresión lineal por columna
model = LinearRegression()
for col in X.columns:
    y = df[col]
    model.fit(X[[col]], y)
    next_day = np.array([[df[col + "_1"].iloc[-1], df[col + "_2"].iloc[-1], df[col + "_3"].iloc[-1], df[col + "_4"].iloc[-1], df[col + "_5"].iloc[-1], df[col + "_6"].iloc[-1], df[col + "_7"].iloc[-1]]])
    predicted_value = model.predict(next_day)
    print(f"{col}: {predicted_value[0]}")



