"""
==========
b. scatter
==========
"""

from easy_mpl import scatter
import numpy as np
import matplotlib.pyplot as plt


#############################

x_ = np.random.random(100)
y_ = np.random.random(100)
scatter(x_, y_, show=False)

#############################

# show colorbar
scatter(x_, y_, colorbar=True, show=False)

#############################

# retrieve axes for further processing
axes, _ = scatter(x_, y_, show=False)
assert isinstance(axes, plt.Axes)