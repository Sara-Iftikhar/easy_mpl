
import unittest

import matplotlib.pyplot as plt
import numpy as np

from easy_mpl import surf


class TestSurf(unittest.TestCase):

    show = False

    X = np.random.random(100)
    Y = np.random.random(100)
    Z = np.random.random(100)
    C = np.random.random(100)

    def test_with_xy(self):

        ax = surf(self.X, self.Y, show=self.show)
        assert isinstance(ax, plt.Axes)

        return

    def test_with_xyz(self):

        ax = surf(self.X, self.Y, self.Z, show=self.show)
        assert isinstance(ax, plt.Axes)

        return

    def test_with_xyzc(self):

        ax = surf(self.X, self.Y, self.Z, self.C, show=self.show)
        assert isinstance(ax, plt.Axes)

        return


if __name__ == "__main__":
    unittest.main()
