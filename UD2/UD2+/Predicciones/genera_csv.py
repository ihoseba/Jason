import pandas as pd
import numpy as np

# Crear fechas mensuales para 10 años
dates = pd.date_range(start='2023-01-01', periods=120, freq='M')

# Generar datos aleatorios (10%)
random_data = np.random.rand(12) * 100

# Generar datos con incremento lineal (90%)
linear_data = np.linspace(100, 200, 108)

# Combinar datos aleatorios y lineales
all_data = np.concatenate([random_data, linear_data])

# Crear DataFrame con fechas y datos
df = pd.DataFrame({'Date': dates, 'SMA_30': all_data, 'Close': all_data + np.random.normal(0, 5, len(all_data))})

# Escribir en un archivo CSV
df.to_csv('mi_archivo.csv', index=False)

print("Datos guardados en mi_archivo.csv")
