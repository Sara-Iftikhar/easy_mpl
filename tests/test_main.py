
import random
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import bar_chart, imshow, hist, pie, plot
from easy_mpl import regplot, scatter, contour
from easy_mpl.utils import BAR_CMAPS, get_cmap
from easy_mpl import dumbbell_plot, ridge
from easy_mpl import parallel_coordinates
from easy_mpl import lollipop_plot
from easy_mpl import circular_bar_plot



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
    
    def test_with_nan_vals(self):
        ax = bar_chart(values=[1, 2, np.nan, 4, 5], show=self.show,
            title='test_with_nan_vals')
        assert isinstance(ax, plt.Axes)
        return
    
    def test_figsize(self):
        ax = bar_chart(values=[1, 2, 3, 4, 5], 
            figsize=(10, 10),
            show=self.show,
            title='test_with_nan_vals')
        assert isinstance(ax, plt.Axes)
        return

    def test_err_h(self):
        x = np.random.randint(1, 10, 10)
        err = np.random.random(10)
        ax = bar_chart(x, errors=err, orient="v",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_err_v(self):
        x = np.random.randint(1, 10, 10)
        err = np.random.random(10)
        ax = bar_chart(x, errors=err, orient="h",
                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

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


class TestPlot(unittest.TestCase):
    show = False

    def test_vanilla(self):
        ax = plot(np.random.random(100), title="vanilla", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        ax = plot(np.random.random(100), title="figsize", 
        figsize=(10, 10),
        show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_1array(self):
        ax = plot(np.random.random(100), '.', title="1array", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_color(self):
        ax = plot(np.random.random(100), '.', title="1array",
                  c=np.array([35, 81, 53]) / 256.0,
                  show=self.show)
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
        ax = plot(np.arange(100), np.random.random(100), '.', title="2array_marker",
                  show=self.show)
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
        ax = plot(np.arange(10), '--', linewidth=1., title="linewidth",
                  show=self.show)
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
        
    def test_with_nan_vals(self):
        x = np.append(np.random.random(100), np.nan)
        ax = plot(x, '.', title="with_nan_vals", show=self.show)
        assert isinstance(ax, plt.Axes)
        return


class TestImshow(unittest.TestCase):
    show = False

    def test_imshow(self):
        ax, _ = imshow(np.random.random((10, 10)), colorbar=True, title="vanilla",
                       show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        ax, _ = imshow(np.random.random((10, 10)), colorbar=True, 
            title="figsize", figsize=(10, 10),
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
    
    def test_with_nan_vals(self):
        x = np.random.random((10, 10))
        x.ravel()[np.random.choice(x.size, 5, replace=False)] = np.nan
        ax, img = imshow(x, colorbar=True, show=self.show, title="with_nan_vals")
        assert isinstance(ax, plt.Axes)
        return


class Testhist(unittest.TestCase):
    show = False

    def test_hist(self):
        hist(np.random.random((10, 1)), show=self.show)
        return

    def test_figsize(self):
        hist(np.random.random((10, 1)), 
        figsize=(10, 10),
        show=self.show)
        return

    def test_hist_with_axes(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 1)), ax=ax, show=self.show)
        return
    
    def test_with_nan_vals(self):
        x = np.random.random((10, 1))
        x.ravel()[np.random.choice(x.size, 5, replace=False)] = np.nan
        ax = hist(x, show=self.show, title="with_nan_vals")
        assert isinstance(ax, plt.Axes)
        return


class TestPie(unittest.TestCase):
    show = False
    save = False

    def test_binary(self):
        ax = pie(np.random.randint(0, 2, 100), show=self.show, save=self.save)
        assert isinstance(ax, plt.Axes)
        return

    def test_multiclass(self):
        ax = pie(np.random.randint(0, 5, 100), show=self.show, save=self.save)
        assert isinstance(ax, plt.Axes)
        return

    def test_string(self):
        ax = pie(['a'] * 60 + ['b'] * 50, show=self.show, save=self.save)
        assert isinstance(ax, plt.Axes)
        return

    def test_fraction(self):
        ax = pie([0.1, 0.2, 0.5, 0.2], show=self.show, save=self.save)
        assert isinstance(ax, plt.Axes)
        return
    
    def test_nan_in_fraction(self):
        ax = pie([0.1, 0.2, np.nan, 0.2], show=self.show, save=self.save,
                 title="nan_in_fraction")
        assert isinstance(ax, plt.Axes)
        return
    
    def test_nan_in_vals(self):
        vals = np.random.randint(0, 5, 100).astype("float32")
        vals[np.random.choice(vals.size, 5, replace=False)] = np.nan
        ax = pie(vals, show=self.show, save=self.save, title="nan_in_vals")
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
    
    def test_array_with_nan_vals(self):
        x = np.random.random(100,).astype("float32")
        x[np.random.randint(0, 100, 10)] = np.nan
        axis = ridge(x, title="array_with_nan_vals", show=self.show)
        for ax in axis:
            assert isinstance(ax, plt.Axes)
        return


class TestParallelCoord(unittest.TestCase):
    show = False
    ynames = ['P1', 'P2', 'P3', 'P4', 'P5']

    N1, N2, N3 = 10, 5, 8
    N = N1 + N2 + N3

    y1 = np.random.uniform(0, 10, N) + 7
    y2 = np.sin(np.random.uniform(0, np.pi, N))
    y3 = np.random.binomial(300, 1 / 10, N)
    y4 = np.random.binomial(200, 1 / 3, N)
    y5 = np.random.uniform(0, 800, N)

    data = np.column_stack((y1, y2, y3, y4, y5))
    data = pd.DataFrame(data, columns=ynames)

    def test_catfeatures(self):
        data = self.data.copy()
        data['P5'] = random.choices(['a', 'b', 'c', 'd', 'e', 'f'], k=self.N)
        ax = parallel_coordinates(data, names=self.ynames, title="cat feature",
                             show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_customticklabelscatfeatures(self):
        data = self.data.copy()
        data['P5'] = random.choices(['a', 'b', 'c', 'd', 'e', 'f'], k=self.N)
        ax = parallel_coordinates(data, title="custom ticklabels cat feat",
                                  ticklabel_kws={"fontsize": 8, "color": "red"},
                                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_allcontinuous(self):
        data = self.data.copy()
        ax = parallel_coordinates(data, names=self.ynames, title="all continuous",
                             show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_continuoustarget(self):
        data = self.data.copy()
        ax = parallel_coordinates(data, names=self.ynames,
                                  categories=np.random.randint(0, 5, self.N),
                                  title="target continuous",
                                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_targetcategory(self):
        category = ['a', 'b', 'c', 'd', 'e']
        data = self.data.copy()
        ax = parallel_coordinates(data, names=self.ynames,
                                  categories=random.choices(category, k=len(data)),
                                  show=self.show,
                         title="target category")
        assert isinstance(ax, plt.Axes)
        return

    def test_nparray(self):
        data = self.data.copy()
        ax = parallel_coordinates(data.values, names=self.ynames,
                             title="np array",
                             show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_customticklabels(self):
        data = self.data.copy()
        ax = parallel_coordinates(data.values,
                             title="custom ticklabels",
                             ticklabel_kws={"fontsize": 8, "color": "red"},
                             show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_straightlines(self):
        data = self.data.copy()
        ax = parallel_coordinates(data, linestyle="straight",
                             title="straight lines",
                             show=self.show)
        assert isinstance(ax, plt.Axes)
        return
    
    def test_all_cat(self):
        data = {
            'tide': ['yeo', 'scale', 'log', 'minmax', 'robust'],
            'wat': ['scale', 'log', 'sqrt', 'quantile', 'log'],
            'estimator': ['log', 'zscore', 'log', 'robust', 'zscore'],
            'P4': ['lr', 'lasso', 'lr', 'lr', 'lasso'],
        }
        cat = [3.10e14, 1.15e14, 1.20e14, 1.25e14, 1.50e14]

        data = pd.DataFrame.from_dict(data)

        ax = parallel_coordinates(data, cat, show=self.show, title="all cat")
        assert isinstance(ax, plt.Axes)
        return
    
    def test_all_cat_but_one(self):
        data = {
            'tide': ['yeo', 'scale', 'log', 'minmax', 'robust'],
            'wat': ['scale', 'log', 'sqrt', 'quantile', 'log'],
            'estimator': ['log', 'zscore', 'log', 'robust', 'zscore'],
            'P3': [11,2,33,4, 5],
            'P4': ['lr', 'lasso', 'lr', 'lr', 'lasso'],
        }
        cat = [3.10e14, 1.15e14, 1.20e14, 1.25e14, 1.50e14]

        data = pd.DataFrame.from_dict(data)

        ax = parallel_coordinates(data, cat, show=self.show, title="all cat but one")
        assert isinstance(ax, plt.Axes)
        return
    
    def test_nan_in_data(self):
        data = self.data.copy()
        data.iloc[4, 4] = np.nan
        data.iloc[3, 3] = np.nan
        ax = parallel_coordinates(data, names=self.ynames,
                                  title="nan in data",
                                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return
    
    def test_nan_in_categories(self):
        
        data = self.data.copy()
        categories = np.arange(len(data)).astype("float32")
        categories[4] = np.nan

        ax = parallel_coordinates(data, names=self.ynames,
                                  categories=categories,
                                  title="nan in categories",
                                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return
    
    def test_nan_in_data_and_categroes(self):
        data = self.data.copy()
        data.iloc[4, 4] = np.nan
        data.iloc[3, 3] = np.nan
        categories = np.arange(len(data)).astype("float32")
        categories[6] = np.nan
        ax = parallel_coordinates(data, names=self.ynames,
                                  categories=categories,
                                  title="nan in data and categories",
                                  show=self.show)
        assert isinstance(ax, plt.Axes)
        return


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


class TestCircularBarPlot(unittest.TestCase):
    show = False
    data = np.random.random(50, )
    names = [f"{i}" for i in range(50)]

    def test_basic(self):
        ax = circular_bar_plot(self.data, title="basic", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_color(self):
        ax = circular_bar_plot(self.data, color="#61a4b2", show=self.show,
                         title="custom color")
        assert isinstance(ax, plt.Axes)
        return

    def test_with_names(self):
        ax = circular_bar_plot(self.data, self.names, title="with names",
                               show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_sort(self):
        ax = circular_bar_plot(self.data, self.names, sort=True,
                               title="sort", show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_cmap(self):
        ax = circular_bar_plot(self.data, self.names, color='viridis',
                               show=self.show, title="cmap")
        assert isinstance(ax, plt.Axes)
        return

    def test_minmax_range(self):

        ax = circular_bar_plot(self.data, self.names, min_max_range=(1, 10),
                              label_padding=1, show=self.show, title="minmax range")
        assert isinstance(ax, plt.Axes)
        return

    def test_custom_label_format(self):
        ax = circular_bar_plot(self.data, self.names, label_format='{} {:.4f}',
                               show=self.show, title="custom label format")
        assert isinstance(ax, plt.Axes)
        return

    def test_multiple_values(self):
        # circular_bar_plot(data, names)
        # assert isinstance(ax, plt.Axes)
        return
    
    def test_with_nan_vals(self):
        data = np.random.random(50)
        data[10] = np.nan
        names = [f"{i}" for i in range(50)]
        ax = circular_bar_plot(data, names, show=self.show, title="with nan vals")
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()
