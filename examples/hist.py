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

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']

#############################

data = np.random.randn(1000)
hist(data, hist_kws={'bins':100})

#%%
# setting grid to False
hist(data, hist_kws={'bins':100}, grid=False)

#%%

hist(data, hist_kws={'bins':100, 'color': 'green'})

# %%
hist(df[cols])