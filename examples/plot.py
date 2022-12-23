"""
=======
a. plot
=======
.. currentmodule:: easy_mpl

This file shows the usage of :func:`plot` function.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from easy_mpl import plot

# sphinx_gallery_thumbnail_number = 4

#############################
# A basic plot can be drawn just by providing a sequence of numbers to the ``plot``
# function.

_ = plot(np.random.random(50))

# %%
# We can however set the style of the plot/marker using the second argument.

_ = plot(np.random.random(50), '--.')

#############################
# The color can be specified by making use of ``c`` or ``color``
# argument to ``plot`` function.

x = np.random.randint(2, 10, 10)
_ = plot(x, '--o', color=np.array([35, 81, 53]) / 256.0)

#############################
# We can set the show=False in order to further work the current active axes

x2 = np.random.randint(2, 10, 10)
plot(x, '--o', color=np.array([35, 81, 53]) / 256.0, show=False)
_ = plot(x2, '--*', color=np.array([15, 151, 123]) / 256.0)

#############################
# If we provide to arrays to ``plot``, the first array is used for the horizontal axis.
# In this case, the second argument is not used as marker style

_ = plot(np.arange(50, 100), np.random.random(50), title="2 array")

#############################
# However, when we give just one array, the second argument is interpreted as marker style.

_ = plot(np.random.random(100), '--*', title="1array_marker")

#############################
# When we provide two arrays, the third argument is interpreted as marker style.

_ = plot(np.arange(100), np.random.random(100), '.', title="2 array_marker")

#############################
# The legend can be set by making use of ``label`` argument.

_ = plot(np.random.random(100), '--*', label='1array_marker_label')

#############################
# If we want the y-axis to be on log scale, we can set ``logy`` to True.

_ = plot(np.arange(100), np.random.random(100), '--.', title="logy",
              logy=True)

#############################
# The width of the line can be set using ``lw`` or ``linewidth`` argument.

_ = plot(np.arange(10), '--', linewidth=1., title="linewidth")

#############################
# We can also provide three arrays to ``plot`` function.

x = np.random.random(100)
_ = plot(x, x, x, title="3 arrays")

#############################
# In such a case, the fourth argument is interpreted as marker style.

x = np.random.random(100)
_ = plot(x, x, x, '.', title="3array_with_marker")

#############################
# Instead of numpy array, we can also provide pandas Series

x = pd.Series(np.random.random(100), name="Series",
              index=pd.date_range("20100101", periods=100, freq="D"))
_ = plot(x, '.', title="series")

#############################
# or a pandas DataFrame with 1 column

x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                 index=pd.date_range("20100101", periods=100, freq="D"))
_ = plot(x, '.', title="df_1col")

#############################
# If we provide pandas DataFrame with two columns, both columns are plotted.

x = pd.DataFrame(np.random.random((20, 2)),
                 columns=[f"col_{i}" for i in range(2)],
                 index=pd.date_range("20100101", periods=20, freq="D"))
_ = plot(x, '-o', color=np.array([35, 81, 53]) / 256.0, title="df 2cols")

#############################
# lw

_ = plot(np.random.random(10), marker=".", lw=2, title="lw")

#############################
# The marker size can be set using ``markersize`` or ``ms`` argument.

_ = plot(np.random.random(10), marker=".", markersize=10,
          title="markersize")

#############################
# If the array contains nans, they are simply notplotted

x = np.append(np.random.random(100), np.nan)
_ = plot(x, '.', title="with_nan_vals")

# %%
# The ``plot`` function returns matplotlib Axes object, which can be used for further
# processing.

ax = plot(
    np.random.normal(size=100),
    'o',
    show=False,
    xlabel="Predicted",
    ylabel="Residual",
    markerfacecolor=np.array([225, 121, 144])/256.0,
    markeredgecolor="black", markeredgewidth=0.5,
    xlabel_kws={"fontsize": 14},
    ylabel_kws={"fontsize": 14},
     )

# draw horizontal line on y=0
ax.axhline(0.0)
plt.show()

# %%
# We can also provide an already existing axes to ``plot`` functio

_, ax = plt.subplots()
_ = plot(np.random.random(100), ax=ax)

# %%
# The arguments
data = np.column_stack([
     [3.983,1.82,0.397,-0.54,-1.14,-1.48,-1.68,
           -1.76,-1.80,-1.80,-1.74,-1.63,-1.50,-1.40,
           -1.28,-1.16,-1.10,-1.02,-0.94,-0.87,-0.80,
           -0.73,-0.67,-0.61,-0.56,-0.52,-0.48],
    [4.81, 2.92, 1.73, 0.98, 0.51, 0.21, 0.02,
                 -0.08, -0.16, -0.32, -0.35, -0.38, -0.39,
                 -0.40, -0.41, -0.40, -0.38, -0.35, -0.32,
                 -0.29, -0.25, -0.22, -0.19, -0.16, -0.14,
                 -0.11, -0.09,
                 ]]
)

plot(data[:, 0], '-*', lw=2.0, ms=8, label="Na", show=False)
plot(data[:, 1], '-*', label="Ca", show=False,
     legend_kws = {"loc": "upper center", 'prop':{"weight": "bold", 'size': 14}},
     xlabel="Distance", xlabel_kws={"fontsize": 14, 'weight': "bold"},
     ylabel="Energy", ylabel_kws={"fontsize": 14, 'weight': 'bold'},
     xtick_kws = {'labelsize': 12},
     ytick_kws = {'labelsize': 12},
     )
plt.tight_layout()
plt.show()