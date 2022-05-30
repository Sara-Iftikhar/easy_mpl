"""
=========
bar_chart
=========
"""

from easy_mpl import bar_chart

# sphinx_gallery_thumbnail_number = 3
#############################

bar_chart([1,2,3,4,4,5,3,2,5])

#############################


bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])

#############################


bar_chart([1,2,3,4,4,5,3,2,5],
    ['a','b','c','d','e','f','g','h','i'],
          sort=True)