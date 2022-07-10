"""
=====
plot
=====
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from easy_mpl import plot

# sphinx_gallery_thumbnail_number = 2

#############################

plot(np.random.random(50), '--.')

#############################

# specify color
x = np.random.randint(2, 10, 10)
plot(x, '--o', color=np.array([35, 81, 53]) / 256.0)

#############################

# you can get set the show=False in order to further work the current active axes

x2 = np.random.randint(2, 10, 10)
ax = plot(x, '--o', color=np.array([35, 81, 53]) / 256.0, show=False)
plot(x2, '--*', color=np.array([15, 151, 123]) / 256.0)

#############################

# 2array
plot(np.arange(50), np.random.random(50), title="2darray")

#############################

# 1array with marker
plot(np.random.random(100), '--*', title="1array_marker")

#############################

# 2 array with marker
plot(np.arange(100), np.random.random(100), '.', title="2array_marker")

#############################

# 1array with marker and label
plot(np.random.random(100), '--*', label='1array_marker_label')

#############################

# logy
plot(np.arange(100), np.random.random(100), '--.', title="logy",
              logy=True)

#############################

# linewdith
plot(np.arange(10), '--', linewidth=1., title="linewidth")

#############################

# 3array
x = np.random.random(100)
plot(x, x, x, label="3array", title="3arrays")

#############################

# 3array_with_marker
x = np.random.random(100)
plot(x, x, x, '.', title="3array_with_marker")

#############################

# series
x = pd.Series(np.random.random(100), name="Series",
              index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '.', title="series")

#############################

# df_1col
x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                 index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '.', title="df_1col")

#############################

# df_ncol:
x = pd.DataFrame(np.random.random((100, 2)),
                 columns=[f"col_{i}" for i in range(2)],
                 index=pd.date_range("20100101", periods=100, freq="D"))
plot(x, '-', title="df_ncol")

#############################
# lw
plot(np.random.random(10), marker=".", lw=2, title="lw")

#############################

# markersize
plot(np.random.random(10), marker=".", markersize=10,
          title="markersize")

#############################

# with_nan_vals
x = np.append(np.random.random(100), np.nan)
plot(x, '.', title="with_nan_vals")

#%%
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