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

boxplot(df[cols], fill_color="GnBu", patch_artist=True)
