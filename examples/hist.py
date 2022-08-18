"""
=========
hist
=========
"""

from easy_mpl import hist
import numpy as np

#############################

data = np.random.randn(1000)
hist(data, hist_kws={'bins':100})

#%%
# setting grid to False
hist(data, hist_kws={'bins':100}, grid=False)

#%%

hist(data, hist_kws={'bins':100, 'color': 'green'})