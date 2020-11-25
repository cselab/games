#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

delta = 0.025
x = np.arange(0., 3.0, delta)
y = np.arange(0., 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = X * Y

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 30, colors='k')
# ax.clabel(CS, inline=1, fontsize=10)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

plt.show()
