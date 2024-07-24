# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:17:08 2024

@author: joseangelperez
"""

from scipy.stats import spearmanr

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,2,4,3,5,7,6,8,9,10]

coef_sp, p = spearmanr(x,y)

print(coef_sp, p)

alpha=0.85
if p>alpha:
    print()
else:
    pass
