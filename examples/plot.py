"""
=======
a. plot
=======
.. currentmodule:: easy_mpl

This file shows the usage of :func:`plot` function.
"""

# todo, what happens when we pass three or more arrays!

# sphinx_gallery_thumbnail_number = 5

import numpy as np
import pandas as pd
from easy_mpl import plot
import matplotlib.pyplot as plt
from easy_mpl.utils import AddMarginalPlots
from easy_mpl.utils import version_info

version_info()  # print version information of all the packages being used

#############################
# A basic plot can be drawn just by providing a sequence of numbers to the ``plot``
# function.

x = np.linspace(0, 10, 50)
y = np.sin(x)
_ = plot(y)

# %%
# We can however set the style of the plot/marker using the second argument.

_ = plot(y, '--.')

#############################
# The complete list of available marker styles can be seen `here <https://matplotlib.org/stable/api/markers_api.html>_`
#
#
# The color can be specified by making use of ``c`` or ``color``
# argument to ``plot`` function.

_ = plot(y, '--o', color='darkcyan')
# %%
# You can refer to `this <https://matplotlib.org/stable/gallery/color/named_colors.html>`_
# page to see names of all valid matplotlib color names.
#
# We can explicitly provide rgb values of a color.

_ = plot(y, '--o', color=np.array([35, 81, 53]) / 256.0)

#############################
# We can set the show=False in order to further work the current active axes

y2 = np.cos(x)
plot(y, '--o', color=np.array([35, 81, 53]) / 256.0, show=False)
_ = plot(y2, '--*', color=np.array([15, 151, 123]) / 256.0)

#############################
# If we provide two arrays to ``plot``, the first array is used for the horizontal axis
# and second argument/array is used as corresponding y values.
# In this case, the second argument is not used to define marker style.

_ = plot(x, y)

#############################
# However, when we give just one array, the second argument is interpreted as marker style.

_ = plot(y, '--*')

#############################
# When we provide two arrays, the third argument is interpreted as marker style.

_ = plot(x, y, '.')

#############################
# The legend can be set by making use of ``label`` argument.

_ = plot(y, '--*', label="Sin (x)")

#############################
# If we want the y-axis to be on log scale, we can set ``logy`` to True
# and pass it as ``ax_kws`` dictionary.

_ = plot(x, y+2, '--.',  ax_kws={'logy':True})

#############################
# The width of the line can be set using ``lw`` or ``linewidth`` argument.

_ = plot(y, linewidth=3.)

# %%
_ = plot(y, marker=".", lw=2)

#############################
# Instead of numpy array, we can also provide pandas Series

x = pd.Series(y, name="Series",
              index=pd.date_range("20100101", periods=len(y), freq="D"))
_ = plot(x, '.')

#############################
# or a pandas DataFrame with 1 column

x = pd.DataFrame(y, columns=["sin"],
                 index=pd.date_range("20100101", periods=len(y), freq="D"))
_ = plot(x, '.')

#############################
# It should be noted that the index of pandas Series or DataFrame, which is
# a DateTimeIndex in this case, is used for x-axis
#
# If we provide pandas DataFrame with two columns, both columns are plotted.

x = pd.DataFrame(np.column_stack([y, y2]),
                 columns=["sin", "cos"],
                 index=pd.date_range("20100101", periods=len(y), freq="D"))
_ = plot(x, '-o', color=np.array([35, 81, 53]) / 256.0)

# %%
# For more than one columns, if we don't fix the color, the colors are chosen
# randomly.
dy = np.gradient(y)
dy2 = np.gradient(y2)
x = pd.DataFrame(np.column_stack([y, y2, dy, dy2]),
                 columns=["sin", "cos", "dsin", "dcos"],
                 index=pd.date_range("20100101", periods=len(y), freq="D"))
_ = plot(x, '-o')

# %%
# If the dataframe more than one columne, we can plot each column on separate
# axes

_ = plot(x, '-o', share_axes=False)

#############################
# The marker size can be set using ``markersize`` or ``ms`` argument.

# _ = plot(y, marker=".", markersize=10)

#############################
# If the array contains nans, they are simply notplotted

x = np.append(np.random.random(10), np.nan)
_ = plot(x, '.')

# %%
# The ``plot`` function returns matplotlib Axes object, which can be used for further
# processing.
x = np.random.normal(size=100)
y = np.random.normal(size=100)
e = x-y
ax = plot(
    e,
    'o',
    show=False,
    markerfacecolor=np.array([225, 121, 144])/256.0,
    markeredgecolor="black", markeredgewidth=0.5,
    ax_kws=dict(
    xlabel="Predicted",
    ylabel="Residual",
    xlabel_kws={"fontsize": 14},
    ylabel_kws={"fontsize": 14}),
     )

print(f"Type of ax is: {type(ax)}")

# draw horizontal line on y=0
ax.axhline(0.0)
plt.show()

# %%
# We can add marginal plots to our main plot using ``AddMarginalPlots`` class.
# The marginal plots are used to show the distribution of x-axis data and y-axis data.
# The distribution of x-axis data is shown on top of main plot and the distribution
# of y-axis data is shown on right side of main plot.

ax = plot(
    e,
    'o',
    show=False,
    markerfacecolor=np.array([225, 121, 144])/256.0,
    markeredgecolor="black", markeredgewidth=0.5,
    ax_kws=dict(
    xlabel="Predicted",
    ylabel="Residual",
    xlabel_kws={"fontsize": 14},
    ylabel_kws={"fontsize": 14}),
     )

# draw horizontal line on y=0
ax.axhline(0.0)
AddMarginalPlots(x, y, ax)
plt.show()


# %%
# We can also provide an already existing axes to ``plot`` function using ``ax`` argument.

_, ax = plt.subplots()
_ = plot(np.random.random(100), ax=ax)

# %%
# The arguments for design/manipulation of x/y axis labels and tick labels are handled
# by process_axis function. All the arugments of process_axis function can be given
# to the ``plot`` function.

y1 = [3.983,1.82,0.397,-0.54,-1.14,-1.48,-1.68,
      -1.76,-1.80,-1.80,-1.74,-1.63,-1.50,-1.40,
      -1.28,-1.16,-1.10,-1.02,-0.94,-0.87,-0.80,
      -0.73,-0.67,-0.61,-0.56,-0.52,-0.48]

y2 = [4.81, 2.92, 1.73, 0.98, 0.51, 0.21, 0.02,
      -0.08, -0.16, -0.32, -0.35, -0.38, -0.39,
      -0.40, -0.41, -0.40, -0.38, -0.35, -0.32,
      -0.29, -0.25, -0.22, -0.19, -0.16, -0.14,
      -0.11, -0.09]


plot(y1, '-*', lw=2.0, ms=8, label="Na", show=False)

_ = plot(y2, '-*', label="Ca",
     ax_kws=dict(
     legend_kws = {"loc": "upper center", 'prop':{"weight": "bold", 'size': 14}},
     xlabel="Distance", xlabel_kws={"fontsize": 14, 'weight': "bold"},
     ylabel="Energy", ylabel_kws={"fontsize": 14, 'weight': 'bold'},
     xtick_kws = {'labelsize': 12},
     ytick_kws = {'labelsize': 12}),
     )


# %%
# We can add text to a plot using the axes object returned by the ``plot`` function.

plot(y1, '-*', lw=2.0, ms=8, label="Na", show=False)

ax = plot(y2, '-*', label="Ca", show=False,
          ax_kws=dict(
     legend_kws = {"loc": "upper center", 'prop':{"weight": "bold", 'size': 14}},
     xlabel="Distance", xlabel_kws={"fontsize": 14, 'weight': "bold"},
     ylabel="Energy", ylabel_kws={"fontsize": 14, 'weight': 'bold'},
     xtick_kws = {'labelsize': 12},
     ytick_kws = {'labelsize': 12}),
     )

# Add line conecting mean value and its label
ax.plot([np.argmin(y1), 11], [np.min(y1), 1], ls="dashdot", color="black", zorder=3)

# Add mean value label.
ax.text(np.argmin(y1), 1,
    r"$Na_{\rm{min}} = $" + str(round(np.min(y1), 2)),
    fontsize=13, va="center",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15),
    zorder=10  # to make sure the line is on top
)

# Add line conecting mean value and its label
ax.plot([np.argmin(y2), 17], [np.min(y2), 2], ls="dashdot", color="black", zorder=3)

# Add mean value label.
ax.text(np.argmin(y2), 2,
    r"$Ca_{\rm{min}} = $" + str(round(np.min(y2), 2)),
    fontsize=13, va="center",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15),
    zorder=10  # to make sure the line is on top
)

plt.tight_layout()
plt.show()
