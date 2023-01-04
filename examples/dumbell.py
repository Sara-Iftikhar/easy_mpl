"""
================
g. dumbbell_plot
================
.. currentmodule:: easy_mpl

This file shows the usage of :func:`dumbell` function.
"""

# sphinx_gallery_thumbnail_number = -2

import matplotlib.pyplot as plt
import numpy as np
from easy_mpl import dumbbell_plot
from easy_mpl.utils import version_info

version_info()


#############################
# To plot a dumbbell, we require two arrays of equal length.
st = np.random.randint(1, 5, 10)
en = np.random.randint(11, 20, 10)

_ = dumbbell_plot(st, en)

# %%
# We can sort the dumbbells according the starting value
_ = dumbbell_plot(st, en, sort_start="ascend")

# %%
# The sorting can also be in descending order
_ = dumbbell_plot(st, en, sort_start="descend")

# %%
# We can also sort dumbbells according to end value
_ = dumbbell_plot(st, en, sort_end="ascend")

# %%
# And this can also be in descending order
_ = dumbbell_plot(st, en, sort_end="descend")

#############################

# The properties of line can be modified by providing ``line_kws`` dictionary.
_ = dumbbell_plot(st, en, line_kws={'lw':"2"})

# %%
# We can also specify the line color using a color palette (color map).
_ = dumbbell_plot(st, en, line_color='tab10')

# %%
# The properties of starting and end markers can be modified by making use
# of ``start_kws`` and ``end_kws`` keyword.

_ = dumbbell_plot(
    st, en,
    line_color="Oranges",
    line_kws=dict(lw=5),
    start_kws=dict(s=160, lw=2, zorder=2, color="#242c3c", edgecolors="#242c3c"),
    end_kws=dict(s=200, color="#a6a6a6", edgecolors="#a6a6a6", lw=2.5, zorder=2),

)

# %%
# We can also specifiy a separate color for starting markers. One way of
# doing this is by specifying a palette (colormap) name.
_ = dumbbell_plot(
    st, en,
    line_color="Oranges",
    start_marker_color="Blues",
    line_kws=dict(lw=5),
    start_kws=dict(s=160, lw=2, zorder=2),
    end_kws=dict(s=200, color="#a6a6a6", edgecolors="#a6a6a6", lw=2.5, zorder=2)
)

# %%
# We can also provide colormap for end markers.
_ = dumbbell_plot(
    st, en,
    line_color="Oranges",
    start_marker_color="Blues",
    end_marker_color="Greys",
    line_kws=dict(lw=5),
    start_kws=dict(s=160, lw=2, zorder=2),
    end_kws=dict(s=200, lw=2.5, zorder=2)
)