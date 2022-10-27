"""
============
d. bar_chart
============

.. currentmodule:: easy_mpl

This file shows the usage of :func:`bar_chart` function.
"""

from easy_mpl import bar_chart

# sphinx_gallery_thumbnail_number = 4
#############################

bar_chart([1,2,3,4,4,5,3,2,5])

#############################


bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])

#############################


bar_chart([1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
          sort=True)

#%%

bar_chart(
    [1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
    bar_labels=[11, 23, 12,43, 123, 12, 43, 234, 23],
    cmap="GnBu",
    sort=True)

#%%
bar_chart(
    [1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
    bar_labels=[11, 23, 12,43, 123, 12, 43, 234, 23],
    bar_label_kws={'label_type':'edge'},
    cmap="GnBu",
    sort=True)

#%%
# vertical orientation
bar_chart([1,2,3,4,4,5,3,2,5], orient='v')

#%%
# error bars
errors = [0.1, 0.2, 0.3, 0.24, 0.32, 0.11, 0.32, 0.12, 0.42]
bar_chart([1,2,3,4,4,5,3,2,5], errors=errors)