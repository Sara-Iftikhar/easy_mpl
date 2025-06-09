
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl.utils import create_subplots
from easy_mpl.utils import to_1d_array, has_multi_cols
from easy_mpl.utils import version_info


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

    def test_dict_keys(self):
        x = to_1d_array({'a': 1, 'b': 2}.keys())
        assert x.ndim == 1
        return

    def test_dict_values(self):
        x = to_1d_array({'a': 1, 'b': 2}.values())
        assert x.ndim == 1
        return

    def test_df_index(self):
        x = to_1d_array(pd.DataFrame([1,2,3]).index)
        assert x.ndim == 1
        return


class TestMisc(unittest.TestCase):

    def test_multi_col_1darray(self):
        assert has_multi_cols(np.array([1,2,3])) is False
        return

    def test_multi_col_series(self):
        assert has_multi_cols(pd.Series(np.array([1,2,3]))) is False
        return

    def test_multi_col_1d_df(self):
        assert has_multi_cols(pd.DataFrame(np.array([1,2,3]))) is False
        return

    def test_multi_col_list(self):
        assert has_multi_cols([1,2,3]) is False
        return

    def test_multi_col_2d_df(self):
        assert has_multi_cols(pd.DataFrame(np.array([1,2,3, 4]).reshape(-1, 2)))
        return

    def test_multi_col_2d_array(self):
        assert has_multi_cols(np.array([1,2,3, 4]).reshape(-1, 2))
        return

    def test_version_info(self):
        version_info()
        return


class TestCreateSubplots(unittest.TestCase):

    show = False

    def test_1(self):
        f, axes = create_subplots(1)
        axes.plot([1, 2, 3])
        if self.show:
            plt.show()
        return

    def test_1subplot(self):

        fig, ax = plt.subplots()
        f, axes = create_subplots(1, ax=ax)
        axes.plot([1, 2, 3])
        if self.show:
            plt.show()

        return

    def test_2subplots(self):

        f, axes = create_subplots(2)
        for ax in axes.flat:
            ax.plot([1, 2, 3])
        if self.show:
            plt.show()

        return

    def test_3subplots(self):

        f, axes = create_subplots(3)

        for ax in axes.flat:
            ax.plot([1, 2, 3])

        if self.show:
            plt.show()

        return

    def test_3subplots_sharex(self):

        f, axes = create_subplots(3, sharex="all")

        for ax in axes.flat:
            ax.plot([1, 2, 3])

        if self.show:
            plt.show()

        return

    def test_5subplots(self):

        fig, ax = plt.subplots()
        f, axes = create_subplots(5, ax=ax)
        axes = axes.flat

        for i in range(5):
            ax = axes[i]
            ax.plot([1,2,3])

        if self.show:
            plt.show()

        return

    def test_ncols(self):

        f, axes = create_subplots(5, ncols=2)
        axes = axes.flat

        for i in range(5):
            ax = axes[i]
            ax.plot([1,2,3])

        if self.show:
            plt.show()

    def test_ncols3(self):

        f, axes = create_subplots(5, ncols=3)
        axes = axes.flat

        for i in range(5):
            ax = axes[i]
            ax.plot([1, 2, 3])

        if self.show:
            plt.show()


if __name__ == "__main__":
    unittest.main()