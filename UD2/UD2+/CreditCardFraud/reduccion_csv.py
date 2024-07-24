"""
copilot haz un codigo python que lea un fichero csv y recoja un 1% de los datos de forma aleatoria
el fichero de entrada es creditcard.csv y el fichero de salida creditcard_reduced.csv
"""
import pandas as pd
import random

# Lee el archivo CSV original
filename = "creditcard.csv"
df = pd.read_csv(filename)

# Filtra las filas con 'class' igual a 1
df_class_1 = df[df['Class'] == 1]

# Toma al menos 100 filas con 'class' igual a 1
sample_size = max(100, len(df_class_1))

# Toma un 1% de todas las filas
sample_percent = 0.002
sample = df.sample(frac=sample_percent, random_state=42)

# Combina las filas con 'class' igual a 1 y la muestra aleatoria
final_sample = pd.concat([sample, df_class_1])

# Guarda el resultado en un nuevo archivo CSV
output_filename = "creditcard_reduced+.csv"
final_sample.to_csv(output_filename, index=False)

print(f"Se han guardado {len(final_sample)} filas en {output_filename}.")