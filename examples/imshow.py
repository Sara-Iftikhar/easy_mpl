"""
==========
c. imshow
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`imshow` function.

``imshow`` can be used to draw heatmap of a two dimensional array/data.
"""

# sphinx_gallery_thumbnail_number = 3

import numpy as np
from easy_mpl import imshow
from easy_mpl.utils import version_info

version_info()  # print version information of all the packages being used

#############################

x = np.random.random((10, 8))

_ = imshow(x)

# %%

_ = imshow(x, annotate=True)

#############################
# show colorbar

_ = imshow(x, colorbar=True)

# %%
_ = imshow(x, colorbar=True, cbar_params={"border": False})

# %%
_ = imshow(x, colorbar=True, cbar_params={"border": False, 'pad': 0.4,
                                          "orientation":"horizontal"})

#%%
# Annotation

data = np.random.random((4, 10))

_ = imshow(data, cmap="YlGn",
       xticklabels=[f"Feature {i}" for i in range(data.shape[1])],
       grid_params={'border': True, 'color': 'w', 'linewidth': 2},
           annotate=True,
       colorbar=True)

# %%
# we can specify color of text in each box of imshow for annotation
# For this, ``textcolors`` must a numpy array of shape same as that of data.
# Each value in this numpy array will define color for corresponding box annotation.

data = np.arange(9).reshape((3,3))

_ = imshow(data, cmap="Blues",
       annotate=True,
       annotate_kws={
              "textcolors": np.array([['black', 'black', 'black'],
                                      ['black', 'black', 'black'],
                                     ['white', 'white', 'white']]),
              'fontsize': 14
       },
       colorbar=True)

# %%
# We can decide which portion of heatmap to show using ``mask`` argument

x = np.random.random((20, 20))
imshow(x, mask=True)

# %%
imshow(x, mask="upper")

# %%
imshow(x, mask="lower")

# # %%
# # get axes from im and show its processing