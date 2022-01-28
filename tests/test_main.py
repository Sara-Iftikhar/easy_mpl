
import random
import unittest

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import bar_chart, imshow, hist, pie, plot
from easy_mpl import regplot, scatter, contour
from easy_mpl.utils import BAR_CMAPS, get_cmap
from easy_mpl import dumbbell_plot, ridge


def get_chart_data(n):
    d = np.random.randint(2, 50, n)
    return d, [f'feature_{i}' for i in d]


class TestBarChart(unittest.TestCase):
    show = False
    def test_bar_h(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, ax = plt.subplots()
        bar_chart(values=d, labels=names, ax=ax, color=cm, show=self.show)
        return

    def test_bar_v_without_axis(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, sort=True, show=self.show)

    def test_h_sorted(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, orient='v', show=self.show)
        return

    def test_vertical_without_axis(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)
        bar_chart(values=d, labels=names, color=cm, sort=True, orient='v', show=self.show)
        return

    def test_without_labels(self):
        d = np.random.randint(2, 50, 10)
        bar_chart(values=d, sort=True, show=self.show)
        return


class TestRegplot(unittest.TestCase):
    show = False
    x = np.random.random(100)
    y = np.random.random(100)

    def test_reg_plot_with_line(self):
        regplot(self.x, self.y, ci=None, show=self.show)
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


class TestPlot(unittest.TestCase):
    show = True

    def test_vanilla(self):
        ax = plot(np.random.random(100), title="vanilla", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array(self):
        ax = plot(np.random.random(100), '.', title="1array", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array(self):
        ax = plot(np.arange(100), np.random.random(100), title="2darray",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker(self):
        ax = plot(np.random.random(100), '--*', title="1array_marker",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_2array_marker(self):
        ax = plot(np.arange(100), np.random.random(100), '.', title="2array_marker", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array_marker_label(self):
        ax = plot(np.random.random(100), '--*', label='1array_marker_label',
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_logy(self):
        ax = plot(np.arange(100), np.random.random(100), '--.', title="logy",
                  logy=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_linewdith(self):
        ax = plot(np.arange(10), '--', linewidth=1., title="linewidth", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array(self):
        x = np.random.random(100)
        ax = plot(x, x, x, label="3array", title="3arrays", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_3array_with_marker(self):
        x = np.random.random(100)
        ax = plot(x, x, x, '.', title="3array_with_marker", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_series(self):
        x = pd.Series(np.random.random(100), name="Series",
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', title="series", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_1col(self):
        x = pd.DataFrame(np.random.random(100), columns=["first_col"],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '.', title="df_1col", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_ncol(self):
        x = pd.DataFrame(np.random.random((100, 2)),
                         columns=[f"col_{i}" for i in range(2)],
                      index=pd.date_range("20100101", periods=100, freq="D"))
        ax = plot(x, '-', title="df_ncol", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_lw(self):
        ax = plot(np.random.random(10), marker=".", lw=2, title="lw", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_markersize(self):
        ax = plot(np.random.random(10), marker=".", markersize=10,
                  title="markersize", show=self.show)
        assert isinstance(ax, plt.Axes)
        return


class TestImshow(unittest.TestCase):
    show = False
    def test_imshow(self):
        ax, _ = imshow(np.random.random((10, 10)), colorbar=True, title="vanilla",
                       show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_imshow_witout_cb(self):
        ax, img= imshow(np.random.random((10, 10)), colorbar=False, title="without_colorbar",
                        show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_annotate(self):
        ax, img= imshow(np.random.random((10, 10)), colorbar=False, title="annotate",
                        annotate=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df(self):
        df = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
        ax, img = imshow(df, colorbar=True, show=self.show, title="df")
        assert isinstance(ax, plt.Axes)
        return


class Testhist(unittest.TestCase):
    show = False
    def test_hist(self):
        hist(np.random.random((10, 1)), show=self.show)
        return

    def test_hist_with_axes(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 1)), ax=ax, show=self.show)
        return


class TestPie(unittest.TestCase):
    show = False
    def test_binary(self):
        ax = pie(np.random.randint(0, 2, 100), show=self.show, save=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_multiclass(self):
        ax = pie(np.random.randint(0, 5, 100), show=self.show, save=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_string(self):
        ax = pie(['a'] * 60 + ['b'] * 50, show=self.show, save=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fraction(self):
        ax = pie([0.1, 0.2, 0.5, 0.2], show=self.show, save=self.show)
        assert isinstance(ax, plt.Axes)
        return


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


class TestContour(unittest.TestCase):
    show=False

    npts = 200
    x = np.random.uniform(-2, 2, npts)
    y = np.random.uniform(-2, 2, npts)
    z = x * np.exp(-x**2 - y**2)

    def test_vanilla(self):
        ax = contour(self.x, self.y, self.z, title="vanilla", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between(self):
        ax = contour(self.x, self.y, self.z, fill_between=True, title="fill_between",
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between_without_colorbar(self):
        ax = contour(self.x, self.y, self.z, fill_between=True, colorbar=False,
                     title="fill_between without colorbar", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_fill_between_with_show_points(self):
        ax = contour(self.x, self.y, self.z, fill_between=True, colorbar=True,
                     show_points=True, title="fill_between with show_points",
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_labels(self):
        ax = contour(self.x, self.y, self.z, label_contours=True, title="labels",
                     show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_labels_with_fill_between(self):
        ax = contour(self.x, self.y, self.z, label_contours=True, fill_between=True,
                     title="labels with fill_between", show=self.show)
        assert isinstance(ax, plt.Axes)
        return


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
                           labels=[f'Feature {i}' for i in range(10)])
        assert isinstance(ax, plt.Axes)
        return


class TestRidge(unittest.TestCase):
    show = False

    def test_df_3_cols(self):
        df = pd.DataFrame(np.random.random((100, 3)))
        axis = ridge(df, title="df_3_cols", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_df_20_cols(self):
        df = pd.DataFrame(np.random.random((100, 20)))
        axis = ridge(df, title="df_20_cols", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_df_1col(self):
        df = pd.DataFrame(np.random.random((100, 1)))
        axis = ridge(df, title="df_1col", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_nparray_2d(self):
        axis = ridge(np.random.random((100, 5)), title="nparray_2d", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return

    def test_nparray_1d(self):
        axis = ridge(np.random.random(100,), title="nparray_1d", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()