
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import hist


class Testhist(unittest.TestCase):
    show = False

    def test_hist(self):
        hist(np.random.random((10, 1)), show=self.show)
        return

    def test_figsize(self):
        hist(np.random.random((10, 1)),
        figsize=(10, 10),
        show=self.show)
        return

    def test_hist_with_axes(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 1)), ax=ax, show=self.show)
        return

    def test_with_nan_vals(self):
        x = np.random.random((10, 1))
        x.ravel()[np.random.choice(x.size, 5, replace=False)] = np.nan
        ax = hist(x, show=self.show, title="with_nan_vals")
        assert isinstance(ax, plt.Axes)
        return

    def test_df(self):
        _, ax = plt.subplots()
        hist(pd.DataFrame(np.random.random((10, 2))), ax=ax, show=self.show)
        return

    def test_np_2d(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 2)), ax=ax, show=self.show)
        return

    def test_list(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 2)).tolist(), ax=ax, show=self.show)
        return


if __name__ == "__main__":
    unittest.main()
