"""
=======
imshow
=======
"""

import numpy as np
from easy_mpl import imshow

#############################

x = np.random.random((10, 8))
imshow(x, annotate=True)

#############################

# show colorbar
imshow(x, colorbar=True)