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
import matplotlib.pyplot as plt
from matplotlib import colors
from easy_mpl.utils import version_info


version_info()  # print version information of all the packages being used

# %%

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']

#############################

data = np.random.randn(1000)

# %%
# let's start with a basic histogram

_ = hist(data, bins = 100)



# %%
# adding KDE and specifying line properties

_ = hist(data, add_kde=True, bins=50, color = 'lightpink',
         line_kws={'linestyle': '--',
                   'color': 'darkslategrey',
                   'linewidth': 2.0}
         )

# %%
# setting grid to False

_ = hist(data, bins = 100, grid = False, color = 'c',
         add_kde=True, line_kws={'linestyle': '--',
                                 'color':'firebrick',
                                 'linewidth': 2.0}
         )

# %%
# manipulating kde calculation

_ = hist(data, bins = 100, color = 'gold',
         add_kde=True, kde_kws=dict(cut=0.2),
         line_kws={'linestyle': '--',
                    'color':'firebrick',
                    'linewidth': 2.0})

# %%
# Any argument for matplotlib.hist can be given to hist function for example
# ``color`` or ``edgecolor``

_ = hist(data, bins = 20, linewidth = 0.5,
         edgecolor = "k", grid=False, color='khaki')

# %%
# ``histtype`` defines the type of histogram to show. Here we are
# using ``step`` which generates a lineplot that is by default unfilled.

_ = hist(data, bins = 100, edgecolor = "k", histtype='step')

# %%
# if data contains multiple columns, it will be plotted on same axis

_ = hist(df[cols], alpha=0.7)

# %%
# ``share_axes`` can be set to False to plot multiple columns on different
# axis.

_ = hist(df[cols], share_axes=False,
         bins = 20, linewidth = 0.5, edgecolor = "k", grid=False)

# %%
# Arguments of subplots can be given to ``subplots_kws``

_ = hist(df[cols], share_axes=False, subplots_kws={"sharex": "all"},
         bins = 20, linewidth = 0.5, edgecolor = "k", grid=False)

# %%
# ``return_axes`` can be set to True if we want to further work on
# current axis

outs, axes = hist(df[cols], share_axes=False,
         bins = 20, linewidth = 0.5, edgecolor = "k", grid=False,
                  return_axes=True, show=False)
print(f"{len(outs)} {len(axes)}")
for idx, ax in enumerate(axes):
    ax.set_ylabel('Counts', fontsize=12)
plt.subplots_adjust(wspace=0.35)
plt.show()

# %%
# we can further modify colors for better understanding and correct readings
# for example in this plot, the color represent the frequency of data.

n, bins, patches = hist(data, bins = 20, show=False)

# Setting color
f = ((n ** (1 / 2)) / n.max())
norm = colors.Normalize(f.min(), f.max())

for thisfrac, thispatch in zip(f, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

plt.show()
