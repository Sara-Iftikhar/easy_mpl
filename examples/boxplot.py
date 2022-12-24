"""
============
p. boxplot
============

.. currentmodule:: easy_mpl

This file shows the usage of :func:`boxplot` function.
"""
import pandas as pd

from easy_mpl import boxplot
from easy_mpl.utils import _rescale
from easy_mpl.utils import version_info

version_info()

# sphinx_gallery_thumbnail_number = 7

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
_ = boxplot(df[cols])

# %%
# We can also provide multiple (numpy) array
_ = boxplot(df[cols].values)

# %%
# The fill color can be specificed using any valid matplotlib cmap
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True)

# %%
# change color of median line
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True,
            medianprops={"color": "black"})

# %%
# show the mean line
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True,
            meanline=True, showmeans=True)

# %%
# customize mean line color
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True,
            meanprops={"color": "black"})

# %%
# show notches
_ = boxplot(df[cols],
        fill_color="GnBu",
        notch=True,
        patch_artist=True,
        medianprops={"color": "black"})

# %%
# don't show outliers
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True, showfliers=False)

# %%
# change circle size of fliers
_ = boxplot(df[cols], fill_color="GnBu", flierprops={"ms": 1.0})

# %%
# don't show whiskers
_ = boxplot(df[cols], fill_color="GnBu", patch_artist=True,
            showfliers=False, whis=0.0)

# %%
# Plot each boxplot on separate axes
_ = boxplot(dataframe[cols], flierprops={"ms": 1.0},
            fill_color="GnBu", patch_artist=True,
            share_axes=False, figsize=(5, 7))

# %%
# make boxplots horizontal
_ = boxplot(dataframe[cols], flierprops={"ms": 1.0},
            fill_color="GnBu", patch_artist=True,
            share_axes=False, vert=False, widths=0.7)