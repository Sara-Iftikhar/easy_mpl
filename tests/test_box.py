import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from easy_mpl import boxplot
from easy_mpl.utils import _rescale

f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
for col in df.columns:
    df[col] = _rescale(df[col].values)

colors = ['pink', 'lightblue', 'lightgreen', 'pink', 'lightblue', 'lightgreen']


class TestBox(unittest.TestCase):
    show = False

    def test_basic_df(self):
        ax, out = boxplot(df[cols], show=self.show)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_basic_np(self):
        ax, out = boxplot(df[cols].values, show=self.show)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_notch(self):
        ax, out = boxplot(df[cols], show=self.show, notch=True)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_fill_color(self):
        ax, out = boxplot(df[cols], show=self.show, fill_color=colors)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_fill_color_as_cmap(self):
        ax, out = boxplot(df[cols], show=self.show, patch_artist=True,
                          fill_color="hot")
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_unequal_array(self):
        x1 = np.random.random(100)
        x2 = np.random.random(90)
        ax, out = boxplot([x1, x2], show=self.show, fill_color=colors)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_linecolor(self):
        ax, out = boxplot(df[cols], show=self.show, line_color="red")
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_width(self):
        ax, out = boxplot(df[cols], show=self.show, line_width=3)
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return

    def test_labels(self):
        data = np.random.random((100, 3))
        ax, out = boxplot(data, show=self.show, labels=['a', 'b', 'c'])
        assert isinstance(ax, plt.Axes), ax
        assert isinstance(out, dict)
        return


class TestShareAxes(unittest.TestCase):
    show = False

    def test_basic_df(self):
        axes, outs = boxplot(df[cols], show=self.show, share_axes=False)
        _assert_list(axes, outs)
        return

    def test_basic_np(self):
        axes, outs = boxplot(df[cols].values, show=self.show,
                             share_axes=False)
        _assert_list(axes, outs)
        return

    def test_notch(self):
        axes, outs = boxplot(df[cols], show=self.show, notch=True,
                             share_axes=False)
        _assert_list(axes, outs)
        return

    def test_fill_color(self):
        axes, outs = boxplot(df[cols], show=self.show, fill_color=colors,
                          share_axes=False)
        _assert_list(axes, outs)
        return

    def test_fill_color_as_cmap(self):
        axes, outs = boxplot(df[cols], show=self.show, patch_artist=True,
                          fill_color="hot", share_axes=False)
        _assert_list(axes, outs)
        return

    def test_unequal_array(self):
        x1 = np.random.random(100)
        x2 = np.random.random(90)
        axes, outs = boxplot([x1, x2], show=self.show, fill_color=colors,
                          share_axes=False)
        _assert_list(axes, outs)
        return

    def test_linecolor(self):
        axes, outs = boxplot(df[cols], show=self.show, line_color="red",
                          share_axes=False)
        _assert_list(axes, outs)
        return

    def test_width(self):
        axes, outs = boxplot(df[cols], show=self.show, line_width=3,
                          share_axes=False)
        _assert_list(axes, outs)

        return

    def test_labels(self):
        data = np.random.random((100, 3))
        axes, outs = boxplot(data, show=self.show, labels=['a', 'b', 'c'],
                share_axes=False)
        _assert_list(axes, outs)
        return


def _assert_list(axes, outs):
    for out in outs:
        assert isinstance(out, dict)
    for ax in axes:
        assert isinstance(ax, plt.Axes)
    return


if __name__ == "__main__":
    unittest.main()
