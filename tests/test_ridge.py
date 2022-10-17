
import random
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import ridge


class TestRidge(unittest.TestCase):
    show = False

    def test_df_3_cols(self):
        df = pd.DataFrame(np.random.random((100, 3)), dtype='object')
        axis = ridge(df, title="df_3_cols", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_df_20_cols(self):
        df = pd.DataFrame(np.random.random((100, 20)))
        axis = ridge(df, title="df_20_cols", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_df_1col(self):
        df = pd.DataFrame(np.random.random((100, 1)))
        axis = ridge(df, title="df_1col", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_nparray_2d(self):
        axis = ridge(np.random.random((100, 5)), title="nparray_2d", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_nparray_1d(self):
        axis = ridge(np.random.random(100,), title="nparray_1d", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_array_with_nan_vals(self):
        x = np.random.random(100,).astype("float32")
        x[np.random.randint(0, 100, 10)] = np.nan
        axis = ridge(x, title="array_with_nan_vals", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_cmap_as_color_name(self):
        axis = ridge(np.random.random(100, ), color="white", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_cmap_as_cmap_name(self):
        axis = ridge(np.random.random(100, ), color="GnBu", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_cmap_as_rgb(self):
        axis = ridge(np.random.random(100, ), color=np.random.random(3),
                     show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_cmap_as_array(self):
        axis = ridge(np.random.random((100, 4)), color=np.random.random((3, 4)),
                     show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_arrays_of_unequal_length(self):
        x1 = np.random.random(100)
        x2 = np.random.random(90)
        axis = ridge([x1, x2], color=np.random.random((3, 2)),
                     show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
