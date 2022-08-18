"""
===========
spider plot
===========
This file shows usage of spider_plot function.
"""

import pandas as pd
from easy_mpl import spider_plot

#%%
values = [-0.2, 0.1, 0.0, 0.1, 0.2, 0.3]
spider_plot(values=values)

#%%
# specifying labels
labels = ['a', 'b', 'c', 'd', 'e', 'f']
spider_plot(values=values, labels=labels)

#%%
# # specifying tick size
spider_plot(values, labels, xtick_kws={'size': 13})

#%%

df = pd.DataFrame.from_dict(
    {'a': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
    'b': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15,'f': 0.25}})

spider_plot(df, xtick_kws={'size': 13})

#%%

df = pd.DataFrame.from_dict(
    {'a': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
     'b': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15, 'f': 0.25},
     'c': {'a': -0.1, 'b': 0.3, 'c': 0.15, 'd': 0.24, 'e': 0.18, 'f': 0.2}})
spider_plot(df, xtick_kws={'size': 13})

#%%
# use polygon frame
spider_plot(values=values, frame="polygon")

#%%
spider_plot(df, xtick_kws={'size': 13}, frame="polygon",
           color=['b', 'r', 'g', 'm'],
            fill_color=['b', 'r', 'g', 'm'])