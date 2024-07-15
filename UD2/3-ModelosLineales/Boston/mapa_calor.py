import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

data_full = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(data_full, columns=column_names)

corr = df.corr()
route_csv_outt='correlations.csv'
corr.to_csv(route_csv_outt)

print("Corr", corr)

head = df.head()

columnas=['INDUS','RM','DIS','LSTAT','MEDV']

sns.pairplot(df[columnas],size=1.5)

sns.displot(df['RM'])
sns.displot(df['LSTAT'],kde=True)
sns.displot(df['MEDV'],kde=True)


vals=df[columnas].values
t_vals=vals.T

matriz_corr=np.corrcoef(t_vals)

mapa_calor=sns.heatmap(matriz_corr,
                       xticklabels=columnas,
                       yticklabels=columnas)

