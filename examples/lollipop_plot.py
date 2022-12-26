"""
=================
f. lollipop_plot
=================
.. currentmodule:: easy_mpl

This file shows the usage of :func:`lollipop_plot` function.
"""

import numpy as np
from easy_mpl import lollipop_plot
from easy_mpl.utils import version_info

version_info()

#############################

y = np.random.randint(0, 10, size=10)
# vanilla lollipop plot
_ = lollipop_plot(y, title="vanilla")

#############################

# use both x and y
_ = lollipop_plot(y, np.linspace(0, 100, len(y)), title="with x and y")

#############################

# use custom line style
_ = lollipop_plot(y, line_style='--', title="with custom linestyle")

#############################

# use custom marker style
_ = lollipop_plot(y, marker_style='D', title="with custom marker style")

#############################

# sort the data points before plotting
_ = lollipop_plot(y, sort=True, title="sort")

#############################

# horzontal orientation of lollipops
y = np.random.randint(0, 20, size=10)
_ = lollipop_plot(y, orientation="horizontal", title="horizontal")