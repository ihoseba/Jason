# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:00:21 2024

@author: joseangelperez
"""

import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear figuras y subgráficos
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, y1)
ax1.set_title("Gráfica 1")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, y2)
ax2.set_title("Gráfica 2")

# Mostrar las figuras
plt.show()