"""
==============
q. violin
==============
"""
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import violin_plot
from easy_mpl.utils import _rescale

# %%

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
for col in df.columns:
    df[col] = _rescale(df[col].values)


violin_plot(df[cols])

# %%
axes = violin_plot(df[cols], show=False)
axes.set_facecolor("#fbf9f4")
plt.show()


# %%
axes = violin_plot(df[cols], show=False, cut=0.1)
axes.set_facecolor("#fbf9f4")
plt.show()