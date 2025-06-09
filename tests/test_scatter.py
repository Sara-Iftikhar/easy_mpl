
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import scatter
from easy_mpl.utils import map_array_to_cmap


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

    def test_colorbar_with_colors(self):
        # todo
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        colors, mapper = map_array_to_cmap(y, "Blues")
        ax, sc = scatter(x, y, color=colors, show=False)
        plt.colorbar(mapper, ax=ax)
        plt.close('all')

        return


if __name__ == "__main__":
    unittest.main()
