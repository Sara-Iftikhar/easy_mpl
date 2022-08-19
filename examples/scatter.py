"""
==========
b. scatter
==========
"""

from easy_mpl import scatter
import numpy as np
import matplotlib.pyplot as plt


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