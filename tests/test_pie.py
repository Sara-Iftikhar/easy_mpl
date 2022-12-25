
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import pie


class TestPie(unittest.TestCase):
    show = False

    def test_binary(self):
        ax = pie(np.random.randint(0, 2, 100), show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_multiclass(self):
        ax = pie(np.random.randint(0, 5, 100), show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_string(self):
        ax = pie(['a'] * 60 + ['b'] * 50, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fraction(self):
        ax = pie([0.1, 0.2, 0.5, 0.2], show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_fraction(self):
        ax = pie([0.1, 0.2, np.nan, 0.2], show=self.show,
                 ax_kws={'title':"nan_in_fraction"})
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_vals(self):
        vals = np.random.randint(0, 5, 100).astype("float32")
        vals[np.random.choice(vals.size, 5, replace=False)] = np.nan
        ax = pie(vals, show=self.show, ax_kws={'title':"nan_in_vals"})
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
