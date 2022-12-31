"""
==========
b. scatter
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`scatter` function.

``scatter`` function is very similar to :func:`plot` function, however
it provides some functionalities which are not available for ``plot`` function.

"""

# sphinx_gallery_thumbnail_number = 4

from easy_mpl import scatter
import numpy as np
import matplotlib.pyplot as plt
from easy_mpl.utils import version_info
from easy_mpl.utils import map_array_to_cmap

version_info()  # print version information of all the packages being used


#############################
x = np.linspace(0, 10, 100)
y = np.sin(x)

_ = scatter(x, y)

# %%
# Instead of drawing all the markers in same color, we can
# make the color to show something useful. Below, the color
# represents ``y`` values.
_ = scatter(x, y, c=y)

# %%
# As the value of ``y`` goes higher, the color of marker becomes yellowish.
# On the other hand, as the value of ``y`` goes lower, the color becomes bluish.
#
# Instead of making the color to show values of `y`, can use another array
# for the color.

z = np.arange(100)
_ = scatter(x, y, c=z)

#############################
# Now the color of marker changes from left to right instead of from bottom to top.
#
# We can show the colorbar by setting the ``colorbar`` to True.

_ = scatter(x, y, c=y, colorbar=True)

#############################
# The function ``scatter`` returns a tuple. The first argument is a maplotlib
# axes which can be used for further processing

axes, _ = scatter(x, y, show=False)
assert isinstance(axes, plt.Axes)

# %%
# We can provide the actual values of rbg as list/array to color/c argument.
colors, _ = map_array_to_cmap(y, "Blues")
_ = scatter(x, y, color=colors, colorbar=True)

# %%
# However, if we show the colorbar, the colorbar in such a case will be
# `wrong <https://stackoverflow.com/q/70634122/5982232>`_.

colors, mapper = map_array_to_cmap(y, "Blues")
ax, sc = scatter(x, y, color=colors, show=False)
plt.colorbar(mapper)  # we must privide the mapper to ``colorbar`` otherwise colorbar will be wrong
plt.show()

#%%
# The properties of the markers can be manipulated.

_ = scatter(x, y, edgecolors='black', linewidth=0.5)

# %%
# We can use any color map to show the marker colors. A complete list of valid matplotlib
# colormaps can be found `here <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
#
# The size of the marker can be tuned using ``s`` keyword. If a single value is provided,
# all markers will be of single size. We can however make each maker of variable size
# by passing an array to ``s`` keyword.
#

time = [1, 2, 3, 4, 5, 7, 5.9, 5.5]
parameters = [100, 200, 300, 400, 500, 350, 450, 800]
performance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.52, 0.76]
year = [2011, 2012, 2013, 2014, 2015, 2016, 2015, 2014]

_ = scatter(time, year, c=performance, s=parameters,
        colorbar=True,
        edgecolors='black', linewidth=1.0,
        cmap="RdBu",
        ax_kws={"xlabel":"Computation Time (hours)", 'ylabel': "Publish Year",
                'xlabel_kws': {"fontsize": 14},'ylabel_kws': {"fontsize": 14},
                'top_spine': False, 'right_spine': False})

# %%
# We can also annotate the markers by providing ``marker_labels`` argument.

time = [1, 2, 3, 4, 5, 7, 5.9, 5.5]
parameters = [100, 200, 300, 400, 500, 350, 450, 800]
performance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.52, 0.76]
year = [2011, 2012, 2013, 2014, 2015, 2016, 2015, 2014],
names = ["A", "B", "C", "D", "E", "F", "G", "H"]

_ = scatter(time, year, c=performance, s=parameters,
        colorbar=True,
        edgecolors='black', linewidth=1.0,
        cmap="RdBu",
        marker_labels=names,
        yoffset=0.2,
        xoffset=0.3,
        ax_kws={"xlabel":"Computation Time (hours)", 'ylabel': "Publish Year",
                'xlabel_kws': {"fontsize": 14},'ylabel_kws': {"fontsize": 14},
                'top_spine': False, 'right_spine': False, 'tight_layout': True})