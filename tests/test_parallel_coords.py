
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import parallel_coordinates


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


if __name__ == "__main__":
    unittest.main()