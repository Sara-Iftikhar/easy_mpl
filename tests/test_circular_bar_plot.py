
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

from easy_mpl import circular_bar_plot

import numpy as np
import matplotlib.pyplot as plt


class TestCircularBarPlot(unittest.TestCase):

    show = False
    data = np.random.random(50, )
    names = [f"{i}" for i in range(50)]

    def test_basic(self):
        ax = circular_bar_plot(self.data, ax_kws={'title':"basic"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_color(self):
        ax = circular_bar_plot(self.data, color="#61a4b2", show=self.show,
                         ax_kws={'title':"custom color"})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_names(self):
        ax = circular_bar_plot(self.data, self.names, ax_kws={'title':"with names"},
                               show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_fs(self):
        ax = circular_bar_plot(self.data, self.names, ax_kws={'title':"custon_fontsize"},
                               text_kws={"fontsize": 16}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_sort(self):
        ax = circular_bar_plot(self.data, self.names, sort=True,
                               ax_kws={'title':"sort"}, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_cmap(self):
        ax = circular_bar_plot(self.data, self.names, color='viridis',
                               show=self.show, ax_kws={'title':"cmap"})
        assert isinstance(ax, plt.Axes)
        return

    def test_minmax_range(self):

        ax = circular_bar_plot(self.data, self.names, min_max_range=(1, 10),
                              label_padding=1, show=self.show, ax_kws={'title':"minmax range"})
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_label_format(self):
        ax = circular_bar_plot(self.data, self.names, label_format='{} {:.4f}',
                               show=self.show, ax_kws={'title':"custom label format"})
        assert isinstance(ax, plt.Axes)
        return

    def test_multiple_values(self):
        # circular_bar_plot(data, names)
        # assert isinstance(ax, plt.Axes)
        return

    def test_ax(self):
        ax = plt.subplot(111, polar=True)
        ax = circular_bar_plot(self.data,
                               ax = ax,
                               ax_kws={'title':"axistent axes"},
                               show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_cbar(self):
        ax = circular_bar_plot(self.data,
                               colorbar=True,
                               ax_kws={'title':"colorbar"},
                               show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_nan_vals(self):
        data = np.random.random(50)
        data[10] = np.nan
        names = [f"{i}" for i in range(50)]
        ax = circular_bar_plot(data, names, show=self.show, ax_kws={'title':"with nan vals"})
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
