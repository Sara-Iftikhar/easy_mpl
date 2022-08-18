"""
=======
10. pie
=======
"""

import numpy as np
from easy_mpl import pie

# sphinx_gallery_thumbnail_number = 3

#############################

pie(np.random.randint(0, 3, 100))

#############################

pie([0.2, 0.3, 0.1, 0.4])

#############################

# to explode 0.3
explode = (0, 0.1, 0, 0, 0)
pie(fractions=[0.2, 0.3, 0.15, 0.25, 0.1], explode=explode)