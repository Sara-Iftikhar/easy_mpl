"""
=========
e. hist
=========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`hist` function.
"""

from easy_mpl import hist
import pandas as pd
import numpy as np
from easy_mpl.utils import version_info

version_info()

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']

#############################

data = np.random.randn(1000)
_ = hist(data, bins = 100)

# %%
# setting grid to False
_ = hist(data, bins = 100, grid=False)

# %%
# Any argument for matplotlib.hist can be given to hist function as hist_kws

_ = hist(data, bins = 20, linewidth = 0.5, edgecolor = "k", grid=False)

# %%

_ = hist(data, bins = 100, color = 'green')

# %%
_ = hist(df[cols])

# %%
_ = hist(df[cols], share_axes=False, subplots_kws={"sharex": "all"})
