

import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import plot


class TestPlot(unittest.TestCase):
    show = False

    def test_vanilla(self):
        ax = plot(np.random.random(100), ax_kws={'title':"vanilla"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        ax = plot(np.random.random(100), ax_kws={'title':"figsize", 'figsize':(10, 10)},

        show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array(self):
        ax = plot(np.random.random(100), '.', ax_kws={'title':"1array"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color(self):
        ax = plot(np.random.random(100), '.', ax_kws={'title':"1array"},
                  c=np.array([35, 81, 53]) / 256.0,
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array(self):
        ax = plot(np.arange(100), np.random.random(100), ax_kws={'title':"2darray"},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker(self):
        ax = plot(np.random.random(100), '--*', ax_kws={'title':"1array_marker"},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array_marker(self):
        ax = plot(np.arange(100), np.random.random(100), '.', ax_kws={'title':"2array_marker"},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker_label(self):
        ax = plot(np.random.random(100), '--*', ax_kws={'label':'1array_marker_label'},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_logy(self):
        ax = plot(np.arange(100), np.random.random(100), '--.',
                  ax_kws={'title':"logy", 'logy':True},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_linewdith(self):
        ax = plot(np.arange(10), '--', linewidth=1., ax_kws={'title':"linewidth"},
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array(self):
        x = np.random.random(100)
        ax = plot(x, x, x, label="3array", ax_kws={'title':"3arrays"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array_with_marker(self):
        x = np.random.random(100)
        ax = plot(x, x, x, '.', ax_kws={'title':"3array_with_marker"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_series(self):
        x = pd.Series(np.random.random(100), name="Series",
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', ax_kws={'title':"series"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_1col(self):
        x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', ax_kws={'title':"df_1col"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_ncol(self):
        x = pd.DataFrame(np.random.random((100, 2)), dtype='object',
                         columns=[f"col_{i}" for i in range(2)],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '-', ax_kws={'title':"df_ncol"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_share_axes_df(self):
        x = pd.DataFrame(np.random.random((100, 5)), dtype='object',
                         columns=[f"col_{i}" for i in range(5)],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '-', ax_kws={'title':"df_ncol"},
                  share_axes=False,
                  show=self.show)
        assert isinstance(ax, list), type(ax)
        return

    def test_share_axes_ndarray(self):
        x = np.random.random((100, 5))
        ax = plot(x, '-', ax_kws={'title': "df_ncol"},
                  share_axes=False,
                  show=self.show)
        assert isinstance(ax, list), type(ax)
        return

    def test_lw(self):
        ax = plot(np.random.random(10), marker=".", lw=2, ax_kws={'title':"lw"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_markersize(self):
        ax = plot(np.random.random(10), marker=".", markersize=10,
                  ax_kws={'title':"markersize"}, show=self.show)
        assert isinstance(ax, plt.Axes)

    def test_with_nan_vals(self):
        x = np.append(np.random.random(100), np.nan)
        ax = plot(x, '.', ax_kws={'title':"with_nan_vals"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return



if __name__ == "__main__":
    unittest.main()
