
import random
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import bar_chart
from easy_mpl.utils import BAR_CMAPS, make_cols_from_cmap


def get_chart_data(n):
    d = np.random.randint(2, 50, n)
    if isinstance(n, tuple):
        return d, [f'Feature {i}' for i in range(len(d))]
    return d, [f'feature_{i}' for i in d]

rng = np.random.default_rng(313)
x = rng.integers(1, 10, 10)
err = rng.random(10)

class TestBarChart(unittest.TestCase):

    show = False

    def test_bar_h(self):
        d, names = get_chart_data(5)
        cm = make_cols_from_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, ax = plt.subplots()
        ax = bar_chart(data=d, labels=names, ax=ax, color=cm,
                       show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_series_without_labels(self):
        data = pd.Series([1,2,3,3,5])
        ax = bar_chart(data, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_without_labels(self):
        data = pd.DataFrame([1,2,3,3,5])
        ax = bar_chart(data, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_without_labels(self):
        d = np.random.randint(2, 50, 10)
        ax = bar_chart(data=d, sort=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_deprecation_warn(self):
        d = np.random.randint(2, 50, 20)
        ax = bar_chart(values=d, sort=True, max_bars=10, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_max_bars(self):
        d = np.random.randint(2, 50, 20)
        ax = bar_chart(data=d, sort=True, max_bars=10, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_nan_vals(self):
        ax = bar_chart(data=[1, 2, np.nan, 4, 5], show=self.show,
            ax_kws={'title':'test_with_nan_vals'})
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        ax = bar_chart(data=[1, 2, 3, 4, 5],
            figsize=(10, 10),
            show=self.show,
            ax_kws={'title':'test_with_nan_vals'})
        assert isinstance(ax, plt.Axes)
        return

    def test_err_h(self):
        ax = bar_chart(x, errors=err, orient="v",
                  show=True)
        assert isinstance(ax, plt.Axes)
        return

    def test_1d_array(self):
        ax = bar_chart(x.reshape(-1,1), errors=err, orient="v",
                  show=True)
        assert isinstance(ax, plt.Axes)
        return

    def test_err_v(self):
        x = np.random.randint(1, 10, 10)
        err = np.random.random(10)
        ax = bar_chart(x, errors=err, orient="h",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_labels(self):
        ax = bar_chart(np.random.randint(1, 10, 10),
                       bar_labels=np.random.randint(1, 10, 10),
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_values_as_dict_values(self):
        d, names = get_chart_data(5)
        data = {k:v for k,v in zip(d, names)}
        ax = bar_chart(data.keys(), data.values(),
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color(self):
        ax = bar_chart(np.random.randint(1, 10, 10),
                       color="Blue", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color_3(self):
        ax = bar_chart(np.random.randint(1, 10, (10, 3)),
                       color=["Blue", 'salmon', 'cadetblue'], show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color_for_each_bar(self):
        ax = bar_chart(np.random.randint(1, 10, (5, 1)),
                       color=["Blue", 'salmon', 'cadetblue', "olive", "k"], show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_cmap(self):
        ax = bar_chart(np.random.randint(1, 10, 10),
                       cmap="GnBu",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_cmap_3(self):
        ax = bar_chart(np.random.randint(1, 10, (10, 3)),
                       cmap=['GnBu', 'PuBu', 'PuBuGn'],
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_legend(self):
        data = np.random.randint(1, 10, (5, 2))
        ax = bar_chart(data, color=['salmon', 'cadetblue'],
                       show=self.show)
        assert  isinstance(ax, plt.Axes)
        return


class TestShareAxesFalse(unittest.TestCase):

    show = False

    def test_bar_h_3(self):
        d, names = get_chart_data((5, 3))
        cm = make_cols_from_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, ax = plt.subplots()
        ax_list = bar_chart(data=d, labels=names, ax=ax, color=cm,
                       show=self.show, share_axes=False)
        for ax in ax_list:
            assert isinstance(ax, plt.Axes)
        return

    def test_bar_v_3(self):
        d, names = get_chart_data((5, 3))
        cm = make_cols_from_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, ax = plt.subplots()
        ax_list = bar_chart(data=d, labels=names, ax=ax, color=cm,
                            show=self.show, share_axes=False,
                            orient='v'
                            )
        for ax in ax_list:
            assert isinstance(ax, plt.Axes)
        return

    def test_h_sorted_3(self):
        d, names = get_chart_data((5, 3))
        cm = make_cols_from_cmap(random.choice(BAR_CMAPS), len(d), 0.2)


        self.assertRaises(AssertionError, bar_chart, data=d, sort=True,
                          share_axes=False, color=cm)
        return

    def test_v_sorted_3(self):
        d, names = get_chart_data((5, 3))
        cm = make_cols_from_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        self.assertRaises(AssertionError, bar_chart, d, share_axes=False, orient='v',
                          sort=True, color=cm)
        return

    def test_color_3(self):
        ax_list = bar_chart(np.random.randint(1, 10, (10, 3)),
                       color=["Blue", 'salmon', 'cadetblue'],
                       show=self.show, share_axes=False)
        for ax in ax_list:
            assert isinstance(ax, plt.Axes)
        return

    def test_cmap_3(self):
        ax_list = bar_chart(np.random.randint(1, 10, (10, 3)),
                       cmap=['GnBu', 'PuBu', 'PuBuGn'],
                  show=self.show, share_axes=False)
        for ax in ax_list:
            assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
