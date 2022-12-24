"""
==========
b. scatter
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`scatter` function.

``scatter`` function is very similar to :func:`plot` function, however
it provides some functionalities which are not available for ``plot`` function.

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

# %%
# Instead of drawing all the markers in same color, we can
# make the color to show something useful. Below, the color
# represents ``y`` values.
scatter(x, y, c=y)

# %%
z = np.arange(100)
scatter(x, y, c=z)

#############################
# show colorbar

scatter(x, y, c=y, colorbar=True)

#############################
# The function ``scatter`` returns a tuple. The first argument is a maplotlib
# axes which can be used for further processing

axes, _ = scatter(x, y, show=False)
assert isinstance(axes, plt.Axes)

#%%
# The properties of the markers can be manipulated.

scatter(x, y, edgecolors='black', linewidth=0.5)

# %%
time = [1, 2, 3, 4, 5, 7, 5.9, 5.5]
parameters = [100, 200, 300, 400, 500, 350, 450, 800]
performance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.52, 0.76]
year = [2011, 2012, 2013, 2014, 2015, 2016, 2015, 2014],
names = ["A", "B", "C", "D", "E", "F", "G", "H"]

scatter(time, year, c=performance, s=parameters,
        colorbar=True,
        edgecolors='black', linewidth=1.0,
        cmap="RdBu",
        marker_labels=names,
        yoffset=0.2,
        xoffset=0.3,
        show=False,
        ax_kws={"xlabel":"Computation Time (hours)", 'ylabel': "Publish Year",
                'xlabel_kws': {"fontsize": 14},'ylabel_kws': {"fontsize": 14},
                'top_spine': False, 'right_spine': False, 'tight_layout': True})
plt.tight_layout()
plt.show()