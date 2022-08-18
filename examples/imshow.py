"""
==========
c. imshow
==========
This notebook shows the usage of ``imshow`` function in easy_mpl library.

``imshow`` can be used to draw heatmap of a two dimensional array/data.
"""

# sphinx_gallery_thumbnail_number = 3

import numpy as np
from easy_mpl import imshow

#############################

x = np.random.random((10, 8))
imshow(x, annotate=True)

#############################
# show colorbar
imshow(x, colorbar=True)

#%%
# Annotation
data = np.random.random((4, 10))

imshow(data, cmap="YlGn",
       xticklabels=[f"Feature {i}" for i in range(data.shape[1])],
       white_grid=True, annotate=True,
       colorbar=True)