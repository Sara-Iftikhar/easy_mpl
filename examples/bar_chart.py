"""
============
d. bar_chart
============

.. currentmodule:: easy_mpl

This file shows the usage of :func:`bar_chart` function.
"""

# sphinx_gallery_thumbnail_number = 4

import numpy as np
from easy_mpl import bar_chart
import matplotlib.pyplot as plt
from easy_mpl.utils import version_info

version_info()  # print version information of all the packages being used

#############################
# A basic chart requires just a list of values to represent as bars.
_ = bar_chart([1,2,3,4,4,5,3,2,5])

# %%
# we can also provide a numpy array instead
_ = bar_chart(np.array([1,2,3,4,4,5,3,2,5]))

#############################
# setting the labels for axis

_ = bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])

#############################
# sorting the bars according to their values

_ = bar_chart([1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
          sort=True)

# %%
# The default color of bars are chosen randomly. We can
# specify the color in many ways, e.g. a single color for all
# bars
_ = bar_chart([1,2,3,4,4,5,3,2,5], color="salmon")

#%%
# adding bar labels
_ = bar_chart(
    [1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
    bar_labels=[11, 23, 12,43, 123, 12, 43, 234, 23],
    cmap="GnBu",
    sort=True)

#%%
# putting bar labels outside the bar
_ = bar_chart(
    [1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
    bar_labels=[11, 23, 12,43, 123, 12, 43, 234, 23],
    bar_label_kws={'label_type':'edge'},
    cmap="GnBu",
    sort=True)

#%%
# vertical orientation
_ = bar_chart([1,2,3,4,4,5,3,2,5], orient='v')

# %%
# define color for each bar individually
_ = bar_chart([1,2,3,4,5], color=['c', 'm', 'y', 'royalblue', 'brown'])

#%%
# error bars
errors = [0.1, 0.2, 0.3, 0.24, 0.32, 0.11, 0.32, 0.12, 0.42]
_ = bar_chart([1,2,3,4,4,5,3,2,5], errors=errors)

# %%
# the function bar_chart returns matplotlib axes which can be
# used for further processing

sv_bar = np.arange(20, 100, 10)
names = [f"Feature {n}" for n in sv_bar]

ax = bar_chart(sv_bar, names,
          bar_labels=sv_bar, bar_label_kws={'label_type':'edge'},
          show=False, sort=True, cmap='summer_r')

print(f"Type of ax is {type(ax)}")

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(xlabel='mean(|SHAP value|)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# multipler bar charts
data = np.random.randint(1, 10, (5, 2))
_ = bar_chart(data, color=['salmon', 'cadetblue'])

# %%
# multipler bar charts on separate axes
data = np.random.randint(0, 10, (5, 2))
_ = bar_chart(data, color=['salmon', 'cadetblue'], share_axes=False)

# %%
# specifying colors for group of bars

colors = {'Asia': '#60AB7B',
          'Europe': '#F9B234',
          'Africa': '#E91B23'}

continents = {'Pakistan': 'Asia', 'Iran': 'Asia', 'Syria': 'Asia',
         'Iraq': 'Asia', 'Lebanon': 'Asia', 'Ireland': 'Europe',
         'Germany': 'Europe', 'Norway': 'Europe', 'Ghana': 'Africa',
         'Egypt': 'Africa', 'Moroco': 'Africa', 'Tunis': 'Africa'}

data = [ 17.5,  21.6,  21.6,  47.7,
         0.2,  15.2,   0.4 ,   1.4,
         60.6,   1.5,  11.8,   6.2]

ax = bar_chart(data, list(continents.keys()),
          bar_labels=data, bar_label_kws={'label_type':'edge',
                                            'fontsize': 10,
                                            'weight': 'bold'},
          show=False, sort=True, color=[colors[val] for val in continents.values()],
          ax_kws=dict(top_spine=False, right_spine=False))

ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')
labels = np.unique(list(continents.values()))
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.7)
ax.set(axisbelow=True)  # Hide the grid behind plot objects
ax.set_facecolor('floralwhite')
plt.legend(handles, labels, loc='lower right')
plt.tight_layout()
plt.show()
