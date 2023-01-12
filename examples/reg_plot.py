"""
==========
h. regplot
==========
.. currentmodule:: easy_mpl

This file shows the usage of :func:`regplot` function.
"""
# sphinx_gallery_thumbnail_number = -2
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
_ = regplot(x, y, marker_color='black')


#%%
# customizing line style
_ = regplot(x, y, line_color='black')

#%%
# customizing fill color
_ = regplot(x, y, fill_color='black')

#%%

_ = regplot(x, y, ci=None, line_color='green')

# %%
# We can show distribution of x and y along the marginals
# This can be done by setting the ``marginals`` keyword to True

RIDGE_LINE_KWS = [{'color': 'k', 'lw': 1.0}, {'color': 'crimson', 'lw': 1.0}]
HIST_KWS = [{'color': 'darkcyan'}, {'color': 'tab:brown'}]

_ = regplot(x, y,
             marker_size = 28,
             marker_color='crimson',
             line_color='k',
             scatter_kws={'edgecolors':'black', 'linewidth':0.5,
                          'alpha': 0.5},
             marginals=True,
             marginal_ax_pad=0.25,
             marginal_ax_size=0.7,
             ridge_line_kws=RIDGE_LINE_KWS,
             hist=True,
             hist_kws=HIST_KWS)

# %%
# Instead of drawing histograms, we can decide to fill the ridges
# drawn by kde lines on marginals.

fill_kws = {
    "alpha": 0.5
}
_ = regplot(x, y,
            marker_size = 28,
            marker_color='crimson',
            line_color='k',
            scatter_kws={'edgecolors':'black', 'linewidth':0.5,
                          'alpha': 0.5},
            marginals=True,
            marginal_ax_pad=0.25,
            marginal_ax_size=0.7,
            ridge_line_kws=RIDGE_LINE_KWS,
            hist=False,
            fill_kws=fill_kws)

# %%
cov = np.array(
    [[1.0, 0.9, 0.7],
     [0.9, 1.2, 0.8],
     [0.7, 0.8, 1.4]]
)
data = rng.multivariate_normal(np.zeros(3),
                               cov, size=100)

ax = regplot(data[:, 0], data[:, 1], line_color='black',
             marker_color='crimson',
             scatter_kws={'edgecolors':'black', 'linewidth':0.5, 'alpha': 0.5},
             show=False)
ax = regplot(data[:, 0], data[:, 2], line_color='green', ax=ax,
             scatter_kws={'edgecolors':'black', 'linewidth':0.5, 'alpha': 0.5},
             show=False)
plt.show()