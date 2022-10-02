
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import regplot


class TestRegplot(unittest.TestCase):
    show = False
    x = np.random.random(100)
    y = np.random.random(100)

    def test_reg_plot_with_line(self):
        regplot(self.x, self.y, ci=None, show=self.show)
        return

    def test_figsize(self):
        regplot(self.x, self.y, ci=None,
        figsize=(10, 10),
        show=self.show)
        return

    def test_regplot_with_line_and_ci(self):
        regplot(self.x, self.y, show=False)
        return

    def test_regplot_with_line_ci_and_annotation(self):
        regplot(self.x, self.y, annotation_key="MSE", annotation_val=0.2,
                show=self.show)
        return

    def test_with_list_as_inputs(self):
        regplot(self.x.tolist(), self.y.tolist(),
                show=self.show)
        return

    def test_with_nan_vals(self):
        x = np.random.random(100)
        x[10] = np.nan
        y = np.random.random(100)
        y[5] = np.nan
        ax = regplot(x, y, show=self.show, title='test_with_nan_vals')
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_x(self):
        x = np.append(self.x, np.nan)
        y = np.append(self.y, 0.5)
        ax = regplot(x, y, show=self.show, title='test_with_nan_vals')
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()