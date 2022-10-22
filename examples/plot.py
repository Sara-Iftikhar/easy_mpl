"""
=======
a. plot
=======
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from easy_mpl import plot

# sphinx_gallery_thumbnail_number = 2

#############################
# A basic plot can be drawn just by providing a sequence of numbers to the ``plot``
# function.

plot(np.random.random(50))

# %%
# We can however set the style of the plot/marker using the second argument.

plot(np.random.random(50), '--.')

#############################
# The color can be specified by making use of ``c`` or ``color`` argument to ``plot`` function.

x = np.random.randint(2, 10, 10)
plot(x, '--o', color=np.array([35, 81, 53]) / 256.0)

#############################
# We can get set the show=False in order to further work the current active axes

x2 = np.random.randint(2, 10, 10)
ax = plot(x, '--o', color=np.array([35, 81, 53]) / 256.0, show=False)
plot(x2, '--*', color=np.array([15, 151, 123]) / 256.0)

#############################
# If we provide to arrays to ``plot``, the first array is used for the horizontal axis.
# In this case, the second argument is not used as marker style

plot(np.arange(50, 100), np.random.random(50), title="2darray")

#############################
# However, when we give just one array, the second argument is interpreted as marker style.

plot(np.random.random(100), '--*', title="1array_marker")

#############################
# When we provide two arrays, the third argument is interpreted as marker style.

plot(np.arange(100), np.random.random(100), '.', title="2array_marker")

#############################
# The legend can be set by making use of ``label`` argument.

plot(np.random.random(100), '--*', label='1array_marker_label')

#############################
# If we want the y-axis to be on log scale, we can set ``logy`` to True.

plot(np.arange(100), np.random.random(100), '--.', title="logy",
              logy=True)

#############################
# The width of the line can be set using ``lw`` or ``linewidth`` argument.

plot(np.arange(10), '--', linewidth=1., title="linewidth")

#############################
# We can also provide three arrays to ``plot`` function.

x = np.random.random(100)
plot(x, x, x, label="3array", title="3arrays")

#############################
# In such a case, the fourth argument is interpreted as marker style.

x = np.random.random(100)
plot(x, x, x, '.', title="3array_with_marker")

#############################
# Instead of numpy array, we can also provide pandas Series

x = pd.Series(np.random.random(100), name="Series",
              index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '.', title="series")

#############################
# or a pandas DataFrame with 1 column

x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                 index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '.', title="df_1col")

#############################
# if we provide pandas DataFrame with two columns, both columns are plotted.

x = pd.DataFrame(np.random.random((100, 2)),
                 columns=[f"col_{i}" for i in range(2)],
                 index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '-', title="df_ncol")

#############################
# lw

plot(np.random.random(10), marker=".", lw=2, title="lw")

#############################
# The marker size can be set using ``markersize`` or ``ms`` argument.

plot(np.random.random(10), marker=".", markersize=10,
          title="markersize")

#############################
# If the array contains nans, they are simply notplotted

x = np.append(np.random.random(100), np.nan)
plot(x, '.', title="with_nan_vals")

#%%
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