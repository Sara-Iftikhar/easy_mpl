
import unittest

import numpy as np
import pandas as pd

from easy_mpl.utils import to_1d_array

class Testto1darray(unittest.TestCase):

    def test_list(self):
        x = to_1d_array([1,2,3,4])
        assert x.ndim == 1
        return

    def test_1d_np(self):
        x = to_1d_array(np.array([1,2,3,4]))
        assert x.ndim == 1
        return

    def test_2d_np(self):
        x = to_1d_array(np.array([1,2,3,4]).reshape(-1,1))
        assert x.ndim == 1
        return

    def test_2d_np1(self):
        self.assertRaises(AssertionError, to_1d_array,
                          np.array([1,2,3,4]).reshape(-1,2))
        return

    def test_series(self):
        x = to_1d_array(pd.Series(np.array([1,2,3,4])))
        assert x.ndim == 1
        return

    def test_df_1col(self):
        x = to_1d_array(pd.DataFrame(np.array([1,2,3,4])))
        assert x.ndim == 1
        return

    def test_df_multicols(self):

        self.assertRaises(AssertionError,
                          to_1d_array,
                          pd.DataFrame(np.array([1, 2, 3, 4]).reshape(-1, 2))
                          )
        return


if __name__ == "__main__":
    unittest.main()