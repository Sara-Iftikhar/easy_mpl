
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import lollipop_plot


class TestLollipopPlot(unittest.TestCase):
    show = False

    y = np.random.randint(0, 10, size=10)

    def test_vanilla(self):
        ax = lollipop_plot(self.y, title="vanilla", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_x_and_y(self):
        ax = lollipop_plot(self.y, np.linspace(0, 100, len(self.y)),
         title="with x and y", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_linestyle(self):

        ax = lollipop_plot(self.y, line_style='--', title="with custom linestyle",
         show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_marker_style(self):

        ax = lollipop_plot(self.y, marker_style='D', title="with custom marker style",
         show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_sort(self):

        ax = lollipop_plot(self.y, title="sort", sort=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_horizontal(self):
        y = np.random.randint(0, 20, size=10)
        ax = lollipop_plot(y, title="horizontal", orientation="horizontal", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_nan_vals(self):
        y = np.random.randint(0, 20, size=10).astype("float32")
        y[3] = np.nan
        ax = lollipop_plot(y, title="with nan vals", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_linecolor_cmap(self):
        _ = lollipop_plot(self.y, line_color="RdBu", show=self.show)


if __name__ == "__main__":
    unittest.main()