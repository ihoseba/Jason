# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:33:31 2024

@author: joseangelperez
"""

import seaborn as sns
from matplotlib import pyplot as plt

iris_conj = sns.load_dataset("iris")


sns.pairplot(iris_conj, hue='species', diag_kind='hist')
plt.show()
