"""
=====================
l. circular_bar_plot
=====================
.. currentmodule:: easy_mpl

This file shows the usage of :func:`circular_bar_plot` function.
"""

import numpy as np
from easy_mpl import circular_bar_plot
from easy_mpl.utils import version_info

version_info()

######################################
# basic

data = np.random.random(50, )

_ = circular_bar_plot(data)

# %%

_ = circular_bar_plot(data, colorbar=True)

# %%

_ = circular_bar_plot(data, color="RdBu", colorbar=True)

# %%
_ = circular_bar_plot(data, sort=True, colorbar=True)

######################################
# with names

names = [f"F{i}" for i in range(len(data))]
_ = circular_bar_plot(data, names)

######################################
# sort values

_ = circular_bar_plot(data, names, sort=True, text_kws={"fontsize": 16})

######################################
# custom color map

_ = circular_bar_plot(data, names, color='viridis')

######################################
# custom min and max range

_ = circular_bar_plot(data, names, min_max_range=(1, 10), label_padding=1)

######################################
# custom label format

_ = circular_bar_plot(data, names, label_format='{} {:.4f}')
