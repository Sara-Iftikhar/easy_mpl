"""
=================
f. lollipop_plot
=================
"""

import numpy as np
from easy_mpl import lollipop_plot

#############################

y = np.random.randint(0, 10, size=10)
# vanilla lollipop plot
lollipop_plot(y, title="vanilla")

#############################

# use both x and y
lollipop_plot(y, np.linspace(0, 100, len(y)), title="with x and y")

#############################

# use custom line style
lollipop_plot(y, line_style='--', title="with custom linestyle")

#############################

# use custom marker style
lollipop_plot(y, marker_style='D', title="with custom marker style")

#############################

# sort the data points before plotting
lollipop_plot(y, sort=True, title="sort")

#############################

# horzontal orientation of lollipops
y = np.random.randint(0, 20, size=10)
lollipop_plot(y, orientation="horizontal", title="horizontal")