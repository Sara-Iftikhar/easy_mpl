"""
=======
regplot
=======
"""


import numpy as np
from easy_mpl import regplot

x_, y_ = np.random.random(100), np.random.random(100)
regplot(x_, y_)
