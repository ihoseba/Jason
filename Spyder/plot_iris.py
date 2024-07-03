# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:00:03 2024

@author: joseangelperez
"""

# !pip install seaborn

from sklearn.datasets import load_iris
from seaborn import lmplot

iris_conj = load_iris(as_frame=True)

df=iris_conj['frame']
df.columns=[
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
    ]

column_names=df.columns.values

op=input('Opcion? 0-n:....?')
if op=='0':
    # Petal
    lmplot(df,x='petal_width',y='petal_length',hue='species',fit_reg=False)
elif op=='1':
    # Sepal
    lmplot(df,x='sepal_width',y='sepal_length',hue='species',fit_reg=False)
elif op=='2':
    # Sepal
    lmplot(df,x='petal_width',y='sepal_width',hue='species',fit_reg=False)
elif op=='3':
    # Sepal
    lmplot(df,x='petal_length',y='sepal_length',hue='species',fit_reg=False)
elif op=='4':
    # Sepal
    lmplot(df,x='sepal_width',y='sepal_length',hue='species',fit_reg=False)
else:
    pass


# lmplot(df,x='petal_width',y='petal_length',hue='species',fit_reg=False,scatter_kws{'s':40,})

# lmplot(df,x='petal_width',y='petal_length',hue='species',fit_reg=False)
# lmplot(df,x='petal_width',y='sepal_length',hue='species',fit_reg=False)
# lmplot(df,x='sepal_width',y='petal_length',hue='species',fit_reg=False)


    