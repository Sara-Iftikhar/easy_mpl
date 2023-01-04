
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
        ax, _, _ = dumbbell_plot(self.st, self.en,
                           ax_kws=dict(title="basic"), show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_with_line_kws(self):
        ax, _, _ = dumbbell_plot(self.st, self.en, show=self.show,
                           ax_kws=dict(title="with_line_kws"), line_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_st_kws(self):
        ax, _, _ = dumbbell_plot(self.st, self.en, show=self.show,
                           ax_kws=dict(title="with_st_kws"), start_kws={'color': 'black'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_end_kws(self):
        ax, _, _ = dumbbell_plot(self.st, self.en, show=self.show,
                           ax_kws=dict(title="with_end_kws"), end_kws={'color': 'red'})
        assert isinstance(ax, plt.Axes)
        return

    def test_with_labels(self):
        ax, _, _ = dumbbell_plot(self.st, self.en, show=self.show,
                           ax_kws=dict(title="with labels"),
                           labels=[f'GradientBoostingRegressor {i}' for i in range(10)])
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st(self):
        st = self.st.copy().astype("float32")
        st[0] = np.nan
        ax, _, _ = dumbbell_plot(st, self.en, show=self.show,
                            ax_kws=dict(title="with labels"))
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_en(self):
        en = self.en.copy().astype("float32")
        en[0] = np.nan
        ax, _, _ = dumbbell_plot(self.st, en, show=self.show,
                            ax_kws=dict(title="with labels"))
        assert isinstance(ax, plt.Axes)
        return

    def test_nan_in_st_and_en(self):
        st = self.st.copy().astype("float32")
        en = self.en.copy().astype("float32")
        st[0] = np.nan
        en[4] = np.nan
        ax, _, _ = dumbbell_plot(st, en, show=self.show,
                            ax_kws=dict(title="with labels"))
        assert isinstance(ax, plt.Axes)
        return


class TestLineColor(unittest.TestCase):
    st = np.random.randint(1, 5, 10)
    en = np.random.randint(11, 20, 10)

    show = False

    def test_rgb(self):
        dumbbell_plot(self.st, self.en, line_color=np.random.random(3),
                      show=self.show)
        return
    def test_rgb_array(self):
        dumbbell_plot(self.st, self.en, line_color=np.random.random(size=(10, 3)),
                      show=self.show)
        return
    def test_color_name(self):
        dumbbell_plot(self.st, self.en, line_color="k",
                      show=self.show)
        return
    def test_color_names(self):
        dumbbell_plot(self.st, self.en,
                      line_color=['k','r','k','r','k','r','k','r','k','r'],
                      show=self.show)
        return
    def test_pallete_name(self):
        dumbbell_plot(self.st, self.en,
                      line_color="Blues",
                      show=self.show)
        return


class TestStMarkerColor(unittest.TestCase):
    st = np.random.randint(1, 5, 10)
    en = np.random.randint(11, 20, 10)

    show = False

    def test_rgb(self):
        dumbbell_plot(self.st, self.en, start_marker_color=np.random.random(3),
                      show=self.show)
        return

    def test_rgb_array(self):
        dumbbell_plot(self.st, self.en, start_marker_color=np.random.random(size=(10, 3)),
                      show=self.show)
        return

    def test_color_name(self):
        dumbbell_plot(self.st, self.en, start_marker_color="k",
                      show=self.show)
        return

    def test_color_names(self):
        dumbbell_plot(self.st, self.en,
                      start_marker_color=['k', 'r', 'k', 'r', 'k', 'r', 'k', 'r', 'k', 'r'],
                      show=self.show)
        return

    def test_pallete_name(self):
        dumbbell_plot(self.st, self.en,
                      start_marker_color="Blues",
                      show=self.show)
        return


class TestEnMarkerColor(unittest.TestCase):
    st = np.random.randint(1, 5, 10)
    en = np.random.randint(11, 20, 10)

    show = False

    def test_rgb(self):
        dumbbell_plot(self.st, self.en, end_marker_color=np.random.random(3),
                      show=self.show)
        return

    def test_rgb_array(self):
        dumbbell_plot(self.st, self.en, end_marker_color=np.random.random(size=(10, 3)),
                      show=self.show)
        return

    def test_color_name(self):
        dumbbell_plot(self.st, self.en, end_marker_color="k",
                      show=self.show)
        return

    def test_color_names(self):
        dumbbell_plot(self.st, self.en,
                      end_marker_color=['k', 'r', 'k', 'r', 'k', 'r', 'k', 'r', 'k', 'r'],
                      show=self.show)
        return

    def test_pallete_name(self):
        dumbbell_plot(self.st, self.en,
                      end_marker_color="Blues",
                      show=self.show)
        return


if __name__ == "__main__":
    unittest.main()

