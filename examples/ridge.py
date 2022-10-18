"""
========
i. ridge
========
"""

import numpy as np
from easy_mpl import ridge
import matplotlib.pyplot as plt

#############################

data_ = np.random.random((100, 3))
ridge(data_)

#############################
# specifying colormap

ridge(data_, color="Blues")

#############################
# The data can also be in the form of pandas DataFrame

import pandas as pd
ridge(pd.DataFrame(data_))

# %%

# if we don't want to fill the ridge, we can specify the color as white

ridge(np.random.random(100), color=["white"])

# %%

# we can draw all the ridges on same axes as below

df = pd.DataFrame(np.random.random((100, 3)), dtype='object')
ridge(df, share_axes=True, fill_kws={"alpha": 0.5})

# %%

# we can also provide an existing axes to plot on
_, ax = plt.subplots()
ridge(df, ax=ax, fill_kws={"alpha": 0.5})

# %%

# The data can also be in the form of list of arrays
x1 = np.random.random(100)
x2 = np.random.random(100)
ridge([x1, x2], color=np.random.random((3, 2)))

# %%

# The length of arrays need not to be constant/same. We can use arrays of different lengths
x1 = np.random.random(100)
x2 = np.random.random(90)
ridge([x1, x2], color=np.random.random((3, 2)))
