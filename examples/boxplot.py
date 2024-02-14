"""
============
p. boxplot
============

.. currentmodule:: easy_mpl

This file shows the usage of :func:`boxplot` function.
"""

# sphinx_gallery_thumbnail_number = -3

import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import boxplot, plot
from easy_mpl.utils import _rescale
from easy_mpl.utils import version_info

version_info()  # print version information of all the packages being used


# %%

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
dataframe = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
df = dataframe.copy()
for col in df.columns:
    df[col] = _rescale(df[col].values)

print(f"Our data has {len(df)} rows and {df.shape[1]} columns")

# %%
# To draw a boxplot we can provide a pandas DataFrame
_ = boxplot(df[cols], fill_color='khaki')

# %%
# We can also provide multiple (numpy) array
_ = boxplot(df[cols].values)

# %%
# We can give the list of arrays

data = df.iloc[:, 0:12]
_ = boxplot([data[col].values for col in data.columns])

# %%
# The fill color can be specificed using any valid matplotlib cmap

_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True)

# %%
_ = boxplot(df[cols], fill_color="thistle", line_width=1.5, patch_artist=True)

# %%
# change color of median line

_ = boxplot(df[cols], fill_color="thistle", patch_artist=True,
            medianprops={"color": "purple"})

# %%
# different color for box boundary and filling the box

_ = boxplot(df[cols], fill_color="Pastel2", patch_artist=True,
            meanline=True, showmeans=True, meanprops={"color": "black"})

# %%
# show notches

_ = boxplot(df[cols],
        fill_color="pink",
        notch=True,
        patch_artist=True,
        medianprops={"color": "black"})

# %%
# don't show outliers

_ = boxplot(df[cols], fill_color="bone", patch_artist=True, showfliers=False,
            medianprops={"color": "gold"})

# %%
# change circle size of fliers

_ = boxplot(df[cols], fill_color="gray", patch_artist=True,notch=True,
            flierprops={"ms": 1.0})

# %%
# edit caps and whiskers properties

_ = boxplot(df[cols], fill_color="Pastel2", patch_artist=True,
            flierprops={"ms": 4.0,
                        "marker": 'o',
                        "color": 'thistle',
                        "alpha":0.8},
            medianprops={"color": "black"},
            capprops={'color':'#7570b3', "linewidth":2},
            whiskerprops={'color':'#7570b3', "linewidth":2})

# %%
# don't show whiskers

_ = boxplot(df[cols], fill_color="Pastel1",
            patch_artist=True, notch=True,
            showfliers=False, whis=0.0)

# %%
# If we want to draw a separate boxplot on each axes, we can set the value
# of ``share_axes`` to False.

_ = boxplot(dataframe[cols], flierprops={"ms": 1.0},
            fill_color="ocean", patch_artist=True,
            share_axes=False, figsize=(5, 7))

# %%
# To draw the horizontal boxplots, we can set ``vert`` to False.

_ = boxplot(dataframe[cols], flierprops={"ms": 1.0},
            fill_color="Set2", patch_artist=True, notch=True,
            medianprops={"color": "black"},
            share_axes=False, vert=False, widths=0.7,
            figsize=(8, 7)
            )

# %%
# The boxplot function returns a tuple. The first argument is the matplotlib axes and
# second value is a dictionary consisting of output from ``axes.boxplot``.

ax, _ = boxplot(df[cols], fill_color="Pastel2", patch_artist=True,
            flierprops={"ms": 4.0,
                        "marker": 's',
                        "markerfacecolor": 'lightcoral',
                        "alpha":0.8
                        },
            medianprops={"color": "black"},
            capprops={'color':'#7570b3', "linewidth":2},
            whiskerprops={'color':'#7570b3', "linewidth":2},
             show=False)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set(
    axisbelow=True,  # Hide the grid behind plot objects
    xlabel='Faetures',
    ylabel='Value',
)
ax.set_facecolor('floralwhite')
plt.show()

# %%
# In order to make grouped boxplots, we can drawo two boxplots on same axes.
# We can specify the position of boxes on the axes using ``positions`` argument.

# Some fake data to plot
A= [[1, 2, 5,],  [7, 2]]
B = [[5, 7, 2, 2, 5], [7, 2, 5]]

ax, _ = boxplot(A, line_color='#D7191C', positions=[1, 2], sym='', widths = 0.6,
             show=False)

_, _ = boxplot(B, line_color="#2C7BB6", positions=[4, 5], sym='', widths = 0.6,
             show=False)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set_xticks([1.5, 4.5])
ax.set_xticklabels(['Group1', 'Group2'])
ax.set(
    axisbelow=True,  # Hide the grid behind plot objects
    xlabel='Faetures',
    ylabel='Value',
)

ax.set_facecolor('floralwhite')

plt.show()

# %%
# join mean of each box through a line

plt.close('all')
ax, _ = boxplot(df[cols], fill_color="thistle",
            patch_artist=True, notch=True,
            boxprops = {"linewidth":1.5,
                        "color":'purple'},
            showmeans=True, meanprops={"markerfacecolor": "black",
                                       "markeredgecolor":'black',
                                        "marker":"o"},
            showfliers=False, whis=0.0,
            show=False)

plot(ax.get_xticks(), df[cols].mean().values,
     color="darkorchid", ax=ax)
