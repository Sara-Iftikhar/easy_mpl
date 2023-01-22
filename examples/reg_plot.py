"""
==========
h. regplot
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`regplot` function.
"""
# sphinx_gallery_thumbnail_number = -3
import matplotlib.pyplot as plt
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

# %%
# We can show distribution of x and y along the marginals
# This can be done by setting the ``marginals`` keyword to True

RIDGE_LINE_KWS = [{'color': 'olive', 'lw': 1.0}, {'color': 'firebrick', 'lw': 1.0}]
HIST_KWS = [{'color': 'khaki'}, {'color': 'salmon'}]

_ = regplot(x, y,
             marker_size = 35,
             marker_color='crimson',
             line_color='k',
             fill_color='k',
             scatter_kws={'edgecolors':'black', 'linewidth':0.5,
                          },
             marginals=True,
             marginal_ax_pad=0.25,
             marginal_ax_size=0.7,
             ridge_line_kws=RIDGE_LINE_KWS,
             hist=True,
             hist_kws=HIST_KWS)

# %%
# Instead of drawing histograms, we can decide to fill the ridges
# drawn by kde lines on marginals.

fill_kws = [{'color': 'thistle'}, {'color': 'lightblue'}]
RIDGE_LINE_KWS = [{'color': 'purple', 'lw': 1.0}, {'color': 'teal', 'lw': 1.0}]

_ = regplot(x, y,
            marker_size = 40,
            marker_color='crimson',
            line_color='k',
            fill_color='k',
            scatter_kws={'edgecolors':'black', 'linewidth':0.5,
                          'alpha': 0.5},
            marginals=True,
            marginal_ax_pad=0.25,
            marginal_ax_size=0.7,
            ridge_line_kws=RIDGE_LINE_KWS,
            hist=False,
            fill_kws=fill_kws)

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
             show=False)
ax = regplot(data[:, 0], data[:, 2], line_color='royalblue', ax=ax,
                marker_color='royalblue', marker_size=35, fill_color='royalblue',
             scatter_kws={'edgecolors':'black', 'linewidth':0.8, 'alpha': 0.8},
             show=False)
plt.show()