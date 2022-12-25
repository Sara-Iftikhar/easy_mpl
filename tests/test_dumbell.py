
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import dumbbell_plot


class TestDumbbell(unittest.TestCase):

    show = False
    st = np.random.randint(1, 5, 10)
    en = np.random.randint(11, 20, 10)

    def test_basic(self):
        ax = dumbbell_plot(self.st, self.en,
                           title="basic", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_line_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_line_kws", line_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_st_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_st_kws", start_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_end_kws(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with_end_kws", end_kws={'color': 'red'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_labels(self):
        ax = dumbbell_plot(self.st, self.en, show=self.show,
                           title="with labels",
                           labels=[f'GradientBoostingRegressor {i}' for i in range(10)])
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st(self):
        st = self.st.copy().astype("float32")
        st[0] = np.nan
        ax = dumbbell_plot(st, self.en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_en(self):
        en = self.en.copy().astype("float32")
        en[0] = np.nan
        ax = dumbbell_plot(self.st, en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st_and_en(self):
        st = self.st.copy().astype("float32")
        en = self.en.copy().astype("float32")
        st[0] = np.nan
        en[4] = np.nan
        ax = dumbbell_plot(st, en, show=self.show,
                            title="with labels")
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()

