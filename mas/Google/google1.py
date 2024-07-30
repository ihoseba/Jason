# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:48:38 2024

@author: joseangelperez

copilot escribe codigo python para lo siguiente coge un csv llamado GOOGL.csv
carga los datos Date,Open,High,Low,Close,Adj Close,Volume
Con Date en Mes/Año
Plotea Open, High,Low,Close, Adj Close sobre Date
Plotea en otra grafica Volume sobre Date

"""

import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("GOOG.csv", parse_dates=["Date"], dayfirst=True)

# Asegurarnos de que la columna 'Date' esté en formato Mes/Año
df["Date"] = df["Date"].dt.strftime("%b/%Y")

import matplotlib.pyplot as plt

# Crear una figura con dos subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico 1: Open, High, Low, Close, Adj Close sobre Date
ax1.plot(df["Date"], df["Open"], label="Open", marker="o")
ax1.plot(df["Date"], df["High"], label="High", marker="s")
ax1.plot(df["Date"], df["Low"], label="Low", marker="^")
ax1.plot(df["Date"], df["Close"], label="Close", marker="x")
ax1.plot(df["Date"], df["Adj Close"], label="Adj Close", marker="D")
ax1.set_ylabel("Price")
ax1.set_title("Stock Prices")
ax1.legend()

# Gráfico 2: Volume sobre Date
ax2.bar(df["Date"], df["Volume"], color="purple")
ax2.set_ylabel("Volume")
ax2.set_title("Trading Volume")

# Ajustar el diseño y mostrar los gráficos
plt.tight_layout()
plt.show()
