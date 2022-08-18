"""
===========
11. contour
===========
"""

from easy_mpl import contour
import numpy as np

#############################

_x = np.random.uniform(-2, 2, 200)
_y = np.random.uniform(-2, 2, 200)
_z = _x * np.exp(-_x**2 - _y**2)
contour(_x, _y, _z, fill_between=True, show_points=True)

#############################

# show contour labels
contour(_x, _y, _z, label_contours=True, show_points=True)