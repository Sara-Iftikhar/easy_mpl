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

# sphinx_gallery_thumbnail_number = -3

# %%

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
for col in df.columns:
    df[col] = _rescale(df[col].values)

print(df.shape)

# %%
# To draw a boxplot we can provide a pandas DataFrame
boxplot(df[cols])

# We can also provide a number array
boxplot(df[cols].values)

# %%
# The fill color can be specificed using any valid matplotlib cmap
boxplot(df[cols], fill_color="GnBu", patch_artist=True)

# %%
# change color of median line
boxplot(df[cols], fill_color="GnBu", patch_artist=True, medianprops={"color": "black"})

# %%
# show the mean line
boxplot(df[cols], fill_color="GnBu", patch_artist=True, meanline=True, showmeans=True)

# %%
# customize mean line color
boxplot(df[cols], fill_color="GnBu", patch_artist=True, meanprops={"color": "black"})

# %%
# show notches
boxplot(df[cols],
        fill_color="GnBu",
        notch=True,
        patch_artist=True,
        medianprops={"color": "black"})

# %%
# don't show outliers
boxplot(df[cols], fill_color="GnBu", patch_artist=True, showfliers=False)

# %%
# don't show whiskers
boxplot(df[cols], fill_color="GnBu", patch_artist=True, showfliers=False, whis=0.0)