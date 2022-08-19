"""
==========
h. regplot
==========
"""


import numpy as np
from easy_mpl import regplot

#%%

rng = np.random.default_rng(313)

x = rng.uniform(0, 10, size=100)
y = x + rng.normal(size=100)
regplot(x, y)


#%%
# customizing marker style
regplot(x, y, marker_color='black')


#%%
# customizing line style
regplot(x, y, line_color='black')

#%%
# customizing fill color
regplot(x, y, fill_color='black')

#%%

regplot(x, y, ci=None, line_color='green')