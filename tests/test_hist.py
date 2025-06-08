
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import hist

rng = np.random.default_rng(313)

data = rng.random((10, 2))
array_1d = data[:, 0]
df = pd.DataFrame(data)


class Testhist(unittest.TestCase):

    show = False

    def test_hist(self):
        hist(np.random.random((10, 1)), show=self.show)
        return

    def test_hist_with_deprecation_warn(self):
        hist(x=np.random.random((10, 1)), show=self.show)
        return

    def test_hist_return_axes(self):
        out, ax = hist(np.random.random((10, 1)), show=self.show,
             return_axes=True)
        assert isinstance(ax, plt.Axes), ax
        return

    def test_hist_return_axes_with_share_axes(self):
        out, ax = hist(np.random.random((10, 2)),
                       show=self.show,
                       share_axes=True,
             return_axes=True)
        assert isinstance(ax, plt.Axes), ax
        return

    def test_hist_return_axes_with_share_axes_False(self):
        out, ax = hist(np.random.random((10, 2)),
                       show=self.show,
                       share_axes=False,
             return_axes=True)
        assert isinstance(ax, np.ndarray), ax
        return

    def test_hist_add_kde(self):
        out = hist(np.random.random((10, 1)), show=self.show,
             add_kde=True)
        assert isinstance(out, tuple)
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
        out = hist(x, show=self.show, ax_kws={'title':"with_nan_vals"})
        assert isinstance(out, tuple)
        return

    def test_df(self):
        hist(df, show=self.show)
        return

    def test_df_1col(self):
        hist(pd.DataFrame(df.iloc[:,0]), show=self.show)
        return

    def test_df_with_ax(self):
        _, ax = plt.subplots()
        hist(pd.DataFrame(np.random.random((10, 2))), ax=ax, show=self.show)
        return

    def test_df_shareax_false(self):
        hist(pd.DataFrame(np.random.random((10, 2))), share_axes=False, show=self.show)
        return

    def test_np_1d(self):
        hist(array_1d, show=self.show)
        return

    def test_np_1d_1(self):
        hist(array_1d.reshape(-1,1),  show=self.show)
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
        hist(data, ax_kws={'title':"labels"}, show=self.show, labels=['a', 'b', 'c'])
        return

    def test_list_of_arrays(self):

        hist([np.random.random(50), np.random.random(50)],
             show=self.show)
        return

    def test_series(self):
        hist(pd.Series(np.random.random(50)),
             show=self.show)
        return

    def test_list_of_series(self):
        hist([pd.Series(np.random.random(50)), pd.Series(np.random.random(50))],
             show=self.show)
        return

    def test_list_of_dfs(self):
        hist([pd.DataFrame(np.random.random(50)), pd.DataFrame(np.random.random(50))],
             show=self.show)
        return 


if __name__ == "__main__":
    unittest.main()
