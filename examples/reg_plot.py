"""
==========
h. regplot
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`regplot` function.
"""
# sphinx_gallery_thumbnail_number = -2

import numpy as np
from easy_mpl import regplot
from easy_mpl.utils import version_info

version_info()

#%%

rng = np.random.default_rng(313)

x = rng.uniform(0, 10, size=100)
y = x + rng.normal(size=100)
_ = regplot(x, y)


#%%
# customizing marker style
_ = regplot(x, y, marker_color='white',
            scatter_kws={'marker':"D", 'edgecolors':'black'})

#%%
# another example by increasing the `marker size`
_ = regplot(x, y, marker_color='crimson', marker_size=35,
           scatter_kws={'marker':"o", 'edgecolors':'black'})


#%%
# customizing line style
_ = regplot(x, y, marker_color='dodgerblue', marker_size=35,
           scatter_kws={'marker':"o"},
            line_color='dimgrey', line_style='--',
            line_kws={'linewidth':3})

#%%
# customizing fill color
_ = regplot(x, y, marker_color='crimson', marker_size=40,
           scatter_kws={'marker':"o", 'edgecolors':'black'},
            fill_color='teal')

#%%
# hiding confidence interval
_ = regplot(x, y, marker_color='crimson', marker_size=40,
           scatter_kws={'marker':"o", 'edgecolors':'black'},
            ci=None, line_color='olive')
