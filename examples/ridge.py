"""
========
i. ridge
========
"""

import numpy as np
from easy_mpl import ridge


#############################

data_ = np.random.random((100, 3))
ridge(data_)

#############################

# specifying colormap
ridge(data_, cmap="Blues")

#############################

# using pandas DataFrame
import pandas as pd
ridge(pd.DataFrame(data_))