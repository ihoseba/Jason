
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos históricos de la acción (reemplaza 'mi_archivo.csv' con tus datos)
df = pd.read_csv('mi_archivo.csv', parse_dates=['Date'], index_col='Date')

# Crear características relevantes (por ejemplo, promedio móvil de 30 días)
df['SMA_30'] = df['Close'].rolling(window=30).mean()

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['SMA_30']]  # Características
y = df['Close']     # Variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones a 10 años (mensuales)
future_dates = pd.date_range(start=df.index[-1], periods=120, freq='M')
future_X = pd.DataFrame({'SMA_30': np.nan}, index=future_dates)
future_y = model.predict(future_X)

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Datos históricos')
plt.plot(future_dates, future_y, label='Predicciones')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre')
plt.title('Predicción de la acción en la bolsa')
plt.legend()
plt.show()
