
import os
import site
import unittest

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
        plt.close('all')
        return

    def test_figsize(self):
        regplot(self.x, self.y, ci=None,
        figsize=(10, 10),
        show=self.show)
        plt.close('all')
        return

    def test_regplot_with_line_and_ci(self):
        regplot(self.x, self.y, show=False)
        plt.close('all')
        return

    def test_regplot_with_label(self):
        regplot(self.x, self.y, label="MSE",
                show=self.show)
        plt.close('all')
        return

    def test_with_list_as_inputs(self):
        regplot(self.x.tolist(), self.y.tolist(),
                show=self.show)
        plt.close('all')
        return

    def test_with_nan_vals(self):
        x = np.random.random(100)
        x[10] = np.nan
        y = np.random.random(100)
        y[5] = np.nan
        ax = regplot(x, y, show=self.show,
                     ax_kws={'title':'test_with_nan_vals'})
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_nan_in_x(self):
        x = np.append(self.x, np.nan)
        y = np.append(self.y, 0.5)
        ax = regplot(x, y, show=self.show,
                     ax_kws={'title':'test_with_nan_vals'})
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_with_single_c(self):
        ax = regplot(self.x, self.y,
                     marker_color=np.array([0.5546875 , 0.7265625 , 0.84765625]),
                show=self.show)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_with_c_as_array(self):
        ax = regplot(self.x, self.y,
                     marker_color=np.random.random((len(self.x), 3)),
                show=self.show)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_linecolor_as_rgb(self):
        regplot(self.x, self.y, line_color=np.random.random(3), show=self.show)
        plt.close('all')
        return

    def test_fillcolor_as_rgb(self):
        regplot(self.x, self.y, fill_color=np.random.random(3), show=self.show)
        plt.close('all')
        return

    def test_hist_on_marginals(self):
        regplot(self.x, self.y, marginals=True, show=self.show)
        plt.close('all')
        return

    def test_ridge_on_marginals(self):
        regplot(self.x, self.y, marginals=True, hist=False, show=self.show)
        plt.close('all')
        return

    def test_two_regplots(self):
        ax = regplot(self.x, self.y, marginals=True, hist=False, show=False)
        regplot(self.x, self.y, marginals=True, hist=False,
                     ax=ax, show=self.show)
        plt.close('all')
        return

    def test_hist_kws(self):
        # make sure that histogram kws are working
        regplot(self.x, self.y, marginals=True, show=self.show,
                hist_kws={'bins': 100, 'color': 'khaki'})
        plt.close('all')
        return

if __name__ == "__main__":
    unittest.main()