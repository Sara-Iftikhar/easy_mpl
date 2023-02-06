"""
=======
j. pie
=======
.. currentmodule:: easy_mpl

This file shows the usage of :func:`pie` function.
"""

import numpy as np
from easy_mpl import pie
import matplotlib.pyplot as plt
from easy_mpl.utils import version_info, map_array_to_cmap

version_info()

# sphinx_gallery_thumbnail_number = 3

#############################

_ = pie(np.random.randint(0, 3, 100))

#############################

_ = pie([0.2, 0.3, 0.1, 0.4])

#############################

# to explode 0.3
explode = (0, 0.1, 0, 0, 0)
_ = pie(fractions=[0.2, 0.3, 0.15, 0.25, 0.1], explode=explode)

# %%
# Specifying colors for each section of pie chart

rng = np.random.default_rng(313)
labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct', 'Nov','Dec']
y = rng.integers(10, 100, 12)
colors, _ = map_array_to_cmap(y, cmap="tab20")
percent = 100.*y/y.sum()

outs = pie(fractions=percent, autopct=None,
           colors=colors, show=False)
patches, texts = outs
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, percent)]

patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                      key=lambda x: x[2],
                                      reverse=True))

plt.legend(patches, labels, bbox_to_anchor=(1.1, 1.),
           fontsize=8)
plt.tight_layout()
plt.show()