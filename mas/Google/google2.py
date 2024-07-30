# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:48:38 2024

@author: joseangelperez

copilot escribe codigo python para lo siguiente coge un csv llamado GOOG.csv
carga los datos Date,Open,High,Low,Close,Adj Close,Volume
Plotea diariamente Open, High,Low,Close, Adj Close en eje y izquierdo y Volume 
en eje y derecho, sobre Date diario en eje x
con lineas continuas finas de colores exepto vlume en barras, Pon leyenda de cada linea

"""

import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("GOOG.csv", parse_dates=["Date"], dayfirst=True)

# Asegurarnos de que la columna 'Date' esté en formato Mes/Año
df["Date"] = df["Date"].dt.strftime("%b/%Y")

import matplotlib.pyplot as plt

# Crear una figura
plt.figure(figsize=(10, 6))

# Gráfico de líneas para Open, High, Low, Close, Adj Close
plt.plot(df["Date"], df["Open"], label="Open", linestyle="-", color="blue")
plt.plot(df["Date"], df["High"], label="High", linestyle="-", color="green")
plt.plot(df["Date"], df["Low"], label="Low", linestyle="-", color="red")
plt.plot(df["Date"], df["Close"], label="Close", linestyle="-", color="purple")
plt.plot(df["Date"], df["Adj Close"], label="Adj Close", linestyle="-", color="orange")

# Configurar el eje y izquierdo
plt.ylabel("Price")
plt.title("Stock Prices")

# Crear un segundo eje y para Volume (en barras)
ax2 = plt.twinx()
ax2.bar(df["Date"], df["Volume"], label="Volume", color="gray", alpha=0.5)
ax2.set_ylabel("Volume")

# Configurar leyendas
plt.legend(loc="upper left")

# Mostrar el gráfico
plt.tight_layout()
plt.show()

