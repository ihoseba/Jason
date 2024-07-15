# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:03:34 2024

@author: MarkelP

"""


import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

data_full = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(data_full, columns=column_names)

corr = df.corr()
routa_csv_out = \
"I:/Contenidos/UD2/3. Modelos lineales/Apuntes/2. Housing Boston/correlations.csv"
corr.to_csv(routa_csv_out)