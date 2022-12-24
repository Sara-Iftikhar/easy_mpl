"""
==========
b. scatter
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`scatter` function.
"""

from easy_mpl import scatter
import numpy as np
import matplotlib.pyplot as plt
from easy_mpl.utils import version_info

version_info()


#############################

x = np.random.random(100)
y = np.random.random(100)
scatter(x, y)

#############################
# show colorbar

scatter(x, y, colorbar=True)

#############################

# retrieve axes for further processing

axes, _ = scatter(x, y, show=False)
assert isinstance(axes, plt.Axes)

#%%

scatter(x, y, edgecolors='black', linewidth=0.5)