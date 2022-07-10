"""
==================
circular_bar_plot
==================
"""

import numpy as np
from easy_mpl import circular_bar_plot

######################################

data = np.random.random(50, )
# basic
circular_bar_plot(data)

######################################

# with names
names = [f"F{i}" for i in range(len(data))]
circular_bar_plot(data, names)

######################################

# sort values
circular_bar_plot(data, names, sort=True, text_kws={"fontsize": 16})

######################################

# custom color map
circular_bar_plot(data, names, color='viridis')

######################################

# custom min and max range
circular_bar_plot(data, names, min_max_range=(1, 10), label_padding=1)

######################################

# custom label format
circular_bar_plot(data, names, label_format='{} {:.4f}')