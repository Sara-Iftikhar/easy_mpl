
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import spider_plot


class TestSpiderPlot(unittest.TestCase):
    values = [-0.2, 0.1, 0.0, 0.1, 0.2, 0.3]
    labels = ['a', 'b', 'c', 'd', 'e', 'f']

    df = pd.DataFrame.from_dict(
        {'a': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
         'b': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15, 'f': 0.25},
         'c': {'a': -0.1, 'b': 0.3, 'c': 0.15, 'd': 0.24, 'e': 0.18, 'f': 0.2}})

    show = False

    def test_basic(self):
        ax = spider_plot(data=self.values, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_specify_tick_labels(self):

        ax = spider_plot(data=self.values,
                         tick_labels=self.labels, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_tick_size(self):

        ax = spider_plot(self.values, self.labels, xtick_kws={'size': 13},
                         show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_polygon_frame_single(self):

        ax = spider_plot(data=self.values, frame="polygon", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_dict(self):

        ax = spider_plot(self.df, xtick_kws={'size': 13}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_dict_polygon(self):

        ax = spider_plot(self.df, xtick_kws={'size': 13}, frame="polygon",
                    color=['b', 'r', 'g'],
                    fill_color=['b', 'r', 'g'],
                         show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2d_np_array(self):
        ax = spider_plot(self.df.values, labels=self.df.columns.tolist(), show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_user_defined_ax(self):
        _, ax = plt.subplots(subplot_kw= dict(projection='polar'))
        ax = spider_plot(data=self.values, ax=ax, show=self.show)
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()