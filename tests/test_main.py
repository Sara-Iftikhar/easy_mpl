
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import plot
from easy_mpl import scatter
from easy_mpl import dumbbell_plot


class TestPlot(unittest.TestCase):
    show = False

    def test_vanilla(self):
        ax = plot(np.random.random(100), title="vanilla", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        ax = plot(np.random.random(100), title="figsize",
        figsize=(10, 10),
        show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array(self):
        ax = plot(np.random.random(100), '.', title="1array", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color(self):
        ax = plot(np.random.random(100), '.', title="1array",
                  c=np.array([35, 81, 53]) / 256.0,
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array(self):
        ax = plot(np.arange(100), np.random.random(100), title="2darray",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker(self):
        ax = plot(np.random.random(100), '--*', title="1array_marker",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array_marker(self):
        ax = plot(np.arange(100), np.random.random(100), '.', title="2array_marker",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker_label(self):
        ax = plot(np.random.random(100), '--*', label='1array_marker_label',
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_logy(self):
        ax = plot(np.arange(100), np.random.random(100), '--.', title="logy",
                  logy=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_linewdith(self):
        ax = plot(np.arange(10), '--', linewidth=1., title="linewidth",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array(self):
        x = np.random.random(100)
        ax = plot(x, x, x, label="3array", title="3arrays", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array_with_marker(self):
        x = np.random.random(100)
        ax = plot(x, x, x, '.', title="3array_with_marker", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_series(self):
        x = pd.Series(np.random.random(100), name="Series",
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', title="series", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_1col(self):
        x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', title="df_1col", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_ncol(self):
        x = pd.DataFrame(np.random.random((100, 2)), dtype='object',
                         columns=[f"col_{i}" for i in range(2)],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '-', title="df_ncol", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_lw(self):
        ax = plot(np.random.random(10), marker=".", lw=2, title="lw", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_markersize(self):
        ax = plot(np.random.random(10), marker=".", markersize=10,
                  title="markersize", show=self.show)
        assert isinstance(ax, plt.Axes)

    def test_with_nan_vals(self):
        x = np.append(np.random.random(100), np.nan)
        ax = plot(x, '.', title="with_nan_vals", show=self.show)
        assert isinstance(ax, plt.Axes)
        return


class TestScatter(unittest.TestCase):
    show = False
    def test_basic(self):
        x = np.random.random(100)
        y = np.random.random(100)
        scatter(x, y, show=self.show)
        return

    def test_return_axes(self):
        x = np.random.random(100)
        y = np.random.random(100)
        ax, _ = scatter(x, y, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_colorbar(self):
        x = np.random.random(100)
        y = np.random.random(100)
        scatter(x, y, colorbar=True, show=self.show)
        return

    def test_with_nan_in_x(self):
        x = np.random.random(100)
        # 5 random values are nan
        x[np.random.choice(x.size, 5, replace=False)] = np.nan
        y = np.random.random(100)
        ax, _ = scatter(x, y, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_nan_in_y(self):
        x = np.random.random(100)
        y = np.random.random(100)
        # 5 random values are nan
        y[np.random.choice(y.size, 5, replace=False)] = np.nan
        ax, _ = scatter(x, y, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_nan_in_x_and_y(self):
        x = np.random.random(100)
        # 5 random values are nan
        x[np.random.choice(x.size, 5, replace=False)] = np.nan
        y = np.random.random(100)
        # 5 random values are nan
        y[np.random.choice(y.size, 5, replace=False)] = np.nan
        ax, _ = scatter(x, y, show=self.show)
        assert isinstance(ax, plt.Axes)
        return


class TestDumbbell(unittest.TestCase):

    show = False
    st = np.random.randint(1, 5, 10)
    en = np.random.randint(11, 20, 10)

    def test_basic(self):
        ax = dumbbell_plot(self.st, self.en,
                           title="basic", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_line_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_line_kws", line_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_st_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_st_kws", start_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_end_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_end_kws", end_kws={'color': 'red'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_labels(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with labels",
                           labels=[f'GradientBoostingRegressor {i}' for i in range(10)])
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st(self):
        st = self.st.copy().astype("float32")
        st[0] = np.nan
        ax = dumbbell_plot(st, self.en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_en(self):
        en = self.en.copy().astype("float32")
        en[0] = np.nan
        ax = dumbbell_plot(self.st, en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st_and_en(self):
        st = self.st.copy().astype("float32")
        en = self.en.copy().astype("float32")
        st[0] = np.nan
        en[4] = np.nan
        ax = dumbbell_plot(st, en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
