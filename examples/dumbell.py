"""
================
g. dumbbell_plot
================
.. currentmodule:: easy_mpl

This file shows the usage of :func:`dumbell` function.
"""

import numpy as np
from easy_mpl import dumbbell_plot
from easy_mpl.utils import version_info

version_info()


#############################


st = np.random.randint(1, 5, 10)
en = np.random.randint(11, 20, 10)
dumbbell_plot(st, en)

#############################

# modify line color
dumbbell_plot(st, en, line_kws={'color':"black"})