"""
============
p. boxplot
============
"""

import pandas as pd

from easy_mpl import boxplot
from easy_mpl.utils import _rescale

# %%

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
for col in df.columns:
    df[col] = _rescale(df[col].values)

print(df.shape)

# %%
# basic boxplot with color pallete
boxplot(df[cols], fill_color="GnBu", patch_artist=True)

# %%
# show the mean line
boxplot(df[cols], fill_color="GnBu", patch_artist=True, meanline=True, showmeans=True)

# %%
# customize median line color
boxplot(df[cols], fill_color="GnBu", patch_artist=True, medianprops={"color": "black"})

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
