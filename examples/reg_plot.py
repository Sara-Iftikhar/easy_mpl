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
import matplotlib.pyplot as plt
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

# %%
# multiple regression lines with customized marker, line
# and fill style

cov = np.array(
    [[1.0, 0.9, 0.7],
     [0.9, 1.2, 0.8],
     [0.7, 0.8, 1.4]]
)
data = rng.multivariate_normal(np.zeros(3),
                               cov, size=100)

ax = regplot(data[:, 0], data[:, 1], line_color='orange',
             marker_color='orange', marker_size=35, fill_color='orange',
             scatter_kws={'edgecolors':'black', 'linewidth':0.8, 'alpha': 0.8},
             show=False, label="data 1")
_ = regplot(data[:, 0], data[:, 2], line_color='royalblue', ax=ax,
                marker_color='royalblue', marker_size=35, fill_color='royalblue',
             scatter_kws={'edgecolors':'black', 'linewidth':0.8, 'alpha': 0.8},
             show=False, label="data 2", ax_kws=dict(legend_kws=dict(loc=(0.1, 0.8))))
plt.show()