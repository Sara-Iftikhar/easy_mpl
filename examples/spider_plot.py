"""
===============
n. spider plot
===============
.. currentmodule:: easy_mpl

This file shows the usage of :func:`spider_plot` function.
"""

import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import spider_plot

#%%
values = [-0.2, 0.1, 0.0, 0.1, 0.2, 0.3]
spider_plot(data=values)

#%%
# specifying labels
labels = ['a', 'b', 'c', 'd', 'e', 'f']
spider_plot(data=values, labels=labels)

#%%
# # specifying tick size
spider_plot(values, labels, xtick_kws={'size': 13})

#%%

df = pd.DataFrame.from_dict(
    {'summer': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
    'winter': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15,'f': 0.25}})

spider_plot(df, xtick_kws={'size': 13})

#%%

df = pd.DataFrame.from_dict(
    {'summer': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
     'winter': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15, 'f': 0.25},
     'automn': {'a': -0.1, 'b': 0.3, 'c': 0.15, 'd': 0.24, 'e': 0.18, 'f': 0.2}})
spider_plot(df, xtick_kws={'size': 13})

#%%
# use polygon frame
spider_plot(data=values, frame="polygon")

#%%
spider_plot(df, xtick_kws={'size': 13}, frame="polygon",
           color=['b', 'r', 'g', 'm'],
            fill_color=['b', 'r', 'g', 'm'])

# %%
# postprocessing of axes

df = pd.DataFrame.from_dict(
    {
        'Hg (Adsorption)': {'1': 97.4, '2': 92.38, '3': 81.2, '4': 73.2, '5': 66.81},
        'Cd (Adsorption)': {'1': 96.2, '2': 91.1, '3': 80.02, '4': 71.55, '5': 64.8},
        'Pb (Adsorption)': {'1': 92.7, '2': 86.3, '3': 78.4, '4': 71.2, '5': 64.4},
        'Hg (Desorption)': {'1': 97.6, '2': 96.5, '3': 94.1, '4': 91.99, '5': 90.0},
        'Cd (Desorption)': {'1': 97.0, '2': 96.2, '3': 94.7, '4': 93.7, '5': 92.5},
        'Pb (Desorption)': {'1': 97.0, '2': 95.8, '3': 93.7, '4': 91.8, '5': 89.9}
     })

spider_plot(df, frame="polygon",
            fill_kws = {"alpha": 0.0},
            xtick_kws={'size': 14, "weight": "bold", "color": "black"},
            show=False,
            leg_kws = {'bbox_to_anchor': (0.90, 1.1)}
            )

plt.gca().set_rmax(100.0)
plt.gca().set_rmin(60.0)
plt.gca().set_rgrids((60.0, 70.0, 80.0, 90.0, 100.0),
                     ("60%", "70%", "80%", "90%", "100%"),
                     fontsize=14, weight="bold", color="black"),
plt.tight_layout()
plt.show()