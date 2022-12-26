"""
=======
j. pie
=======
.. currentmodule:: easy_mpl

This file shows the usage of :func:`pie` function.
"""

import numpy as np
from easy_mpl import pie
from easy_mpl.utils import version_info

version_info()

# sphinx_gallery_thumbnail_number = 3

#############################

_ = pie(np.random.randint(0, 3, 100))

#############################

_ = pie([0.2, 0.3, 0.1, 0.4])

#############################

# to explode 0.3
explode = (0, 0.1, 0, 0, 0)
_ = pie(fractions=[0.2, 0.3, 0.15, 0.25, 0.1], explode=explode)