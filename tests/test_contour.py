
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import contour


class TestContour(unittest.TestCase):
    show=False

    npts = 200
    x = np.random.uniform(-2, 2, npts)
    y = np.random.uniform(-2, 2, npts)
    z = x * np.exp(-x**2 - y**2)

    def test_vanilla(self):
        ax = contour(self.x, self.y, self.z,
                     ax_kws=dict(title="vanilla"), show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between(self):
        ax = contour(self.x, self.y, self.z, fill_between=True,
                     ax_kws=dict(title="fill_between"),
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between_without_colorbar(self):
        ax = contour(self.x, self.y, self.z, fill_between=True,
                     colorbar=False,
                     ax_kws=dict(title="fill_between without colorbar"),
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between_with_show_points(self):
        ax = contour(self.x, self.y, self.z, fill_between=True,
                     colorbar=True,
                     show_points=True,
                     ax_kws=dict(title="fill_between with show_points"),
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_labels(self):
        ax = contour(self.x, self.y, self.z, label_contours=True,
                     ax_kws=dict(title="labels"),
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_labels_with_fill_between(self):
        ax = contour(self.x, self.y, self.z, label_contours=True,
                     fill_between=True,
                     ax_kws=dict(title="labels with fill_between"),
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
