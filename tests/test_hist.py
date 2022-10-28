
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
        subplots_kws={"figsize":(10, 10)},
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
        hist(pd.DataFrame(np.random.random((10, 2))), show=self.show)
        return

    def test_df_with_ax(self):
        _, ax = plt.subplots()
        hist(pd.DataFrame(np.random.random((10, 2))), ax=ax, show=self.show)
        return

    def test_df_shareax_false(self):
        hist(pd.DataFrame(np.random.random((10, 2))), share_axes=False, show=self.show)
        return

    def test_np_2d(self):
        hist(np.random.random((10, 2)), show=self.show)
        return

    def test_np_2d_with_ax(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 2)), ax=ax, show=self.show)
        return

    def test_np_2d_sharex_false(self):
        hist(np.random.random((10, 2)), share_axes=False, show=self.show)
        return

    def test_list(self):
        x = np.random.random((10, 2))
        hist([x[:, 0].tolist(), x[:, 1].tolist()], show=self.show)
        return

    def test_list_with_ax(self):
        _, ax = plt.subplots()
        x = np.random.random((10, 2))
        hist([x[:, 0].tolist(), x[:, 1].tolist()], ax=ax, show=self.show)
        return

    def test_list_sharex_false(self):
        x = np.random.random((10, 2))
        hist([x[:, 0].tolist(), x[:, 1].tolist()], share_axes=False, show=self.show)
        return

    def test_labels(self):
        data = np.random.random((100, 3))
        hist(data, title="labels", show=self.show, labels=['a', 'b', 'c'])
        return


if __name__ == "__main__":
    unittest.main()
