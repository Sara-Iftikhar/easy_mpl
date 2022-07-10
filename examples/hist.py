"""
=========
hist
=========
"""

from easy_mpl import hist
import numpy as np

#############################


hist(np.random.randn(1000), hist_kws={'bins':100})