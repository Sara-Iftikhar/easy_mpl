
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

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

rng = np.random.default_rng(313)
array_1d = rng.random(100)


class TestBox(unittest.TestCase):
    show = False

    def test_basic_df(self):
        ax, out = boxplot(df[cols], show=self.show)
        _assert_output(ax, out)
        return

    def test_basic_np(self):
        ax, out = boxplot(df[cols].values, show=self.show)
        _assert_output(ax, out)
        return

    def test_basic_series(self):
        ax, out = boxplot(df[cols[0]], show=self.show)
        _assert_output(ax, out)
        return

    def test_1array(self):
        ax, out = boxplot(df['tide_cm'].values, show=self.show)
        _assert_output(ax, out)
        return

    def test_1array_with_label(self):
        ax, out = boxplot(df['tide_cm'].values,
                          labels=['tide_cm'],
                          show=self.show)
        _assert_output(ax, out)
        return

    def test_notch(self):
        ax, out = boxplot(df[cols], show=self.show, notch=True)
        _assert_output(ax, out)
        return

    def test_fill_color(self):
        ax, out = boxplot(df[cols], show=self.show,
                          patch_artist=True,
                          fill_color=colors)
        _assert_output(ax, out)
        return

    def test_fill_color_black(self):
        axes, outs = boxplot(df[cols], show=self.show,
                             patch_artist=True,
                             fill_color="k")
        _assert_output(axes, outs)
        return

    def test_fill_color_rgb(self):
        axes, outs = boxplot(df[cols], show=self.show,
                             fill_color=np.random.random(3),
                             patch_artist=True,
                          )
        _assert_output(axes, outs)
        return

    def test_fill_color_as_cmap(self):
        ax, out = boxplot(df[cols], show=self.show, patch_artist=True,
                          fill_color="hot")
        _assert_output(ax, out)
        return

    def test_unequal_array(self):
        x1 = np.random.random(100)
        x2 = np.random.random(90)
        ax, out = boxplot([x1, x2], show=self.show,
                          fill_color=colors[0:2])
        _assert_output(ax, out)
        return

    def test_linecolor(self):
        ax, out = boxplot(df[cols], show=self.show, line_color="red")
        _assert_output(ax, out)
        return

    def test_width(self):
        ax, out = boxplot(df[cols], show=self.show, line_width=3)
        _assert_output(ax, out)
        return

    def test_labels(self):
        data = np.random.random((100, 3))
        ax, out = boxplot(data, show=self.show, labels=['a', 'b', 'c'])
        _assert_output(ax, out)
        return

    def test_labels_df(self):
        data = np.random.random((100, 3))
        ax, out = boxplot(pd.DataFrame(data), show=self.show,
                          labels=['a', 'b', 'c'])
        _assert_output(ax, out)
        return

    def test_1d_array(self):
        ax, out = boxplot(array_1d, show=self.show)
        _assert_output(ax, out)
        return

    def test_1d_array_1(self):
        ax, out = boxplot(array_1d.reshape(-1,1), show=self.show)
        _assert_output(ax, out)
        return

    def test_list_of_arrays(self):
        data = np.random.random(100)
        ax, out = boxplot([data, data, data], show=self.show)
        _assert_output(ax, out)
        return

    def test_list_of_arrays_2dim(self):
        data = np.random.random(100).reshape(-1,1)
        ax, out = boxplot([data, data, data], show=self.show)
        _assert_output(ax, out)
        return

    def test_list_of_series(self):
        data = pd.Series(np.random.random(100))
        ax, out = boxplot([data, data, data], show=self.show)
        _assert_output(ax, out)
        return

    def test_list_of_arrays_with_labels(self):
        data = np.random.random(100)
        ax, out = boxplot([data, data, data], show=self.show,
                          labels=['a', 'b', 'c'])
        _assert_output(ax, out)
        return

    def test_list_of_series_with_labels(self):
        data = pd.Series(np.random.random(100))
        ax, out = boxplot([data, data, data], show=self.show,
                          labels=['a', 'b', 'c'])
        _assert_output(ax, out)
        return

    def test_two_boxplots_on_same_axes(self):
        # Some fake data to plot
        A = [[1, 2, 5, ], [7, 2]]
        B = [[5, 7, 2, 2, 5], [7, 2, 5]]

        boxplot(A, line_color='#D7191C', positions=[1, 2], sym='', widths=0.6,
                          show=False)
        ax, out = boxplot(B, line_color="#2C7BB6", positions=[4, 5], sym='', widths=0.6,
                         show=False)
        _assert_output(ax, out)

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
        axes, outs = boxplot(df[cols], show=self.show,
                             patch_artist=True, fill_color=colors,
                          share_axes=False)
        _assert_list(axes, outs)
        return

    def test_fill_color_black(self):
        axes, outs = boxplot(df[cols], show=self.show,
                             patch_artist=True, fill_color="k",
                          share_axes=False)
        _assert_list(axes, outs)
        return

    def test_fill_color_rgb(self):
        axes, outs = boxplot(df[cols], show=self.show,
                             fill_color=np.random.random(3),
                             patch_artist=True,
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
        axes, outs = boxplot([x1, x2], show=self.show,
                             fill_color=colors[0:2],
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

    def test_labels_df(self):
        data = np.random.random((100, 3))
        axes, outs = boxplot(pd.DataFrame(data), show=self.show,
                             labels=['a', 'b', 'c'],
                share_axes=False)
        _assert_list(axes, outs)
        return

    def test_list_of_arrays(self):
        data = np.random.random(100)
        ax, out = boxplot([data, data, data], show=self.show,
                          share_axes=False)
        _assert_list(ax, out)
        return

    def test_list_of_arrays_2dim(self):
        data = np.random.random(100).reshape(-1,1)
        ax, out = boxplot([data, data, data], show=self.show,
                          share_axes=False)
        _assert_list(ax, out)
        return

    def test_list_of_series(self):
        data = pd.Series(np.random.random(100))
        ax, out = boxplot([data, data, data], show=self.show,
                          share_axes=False)
        _assert_list(ax, out)
        return

    def test_list_of_arrays_with_labels(self):
        data = np.random.random(100)
        ax, out = boxplot([data, data, data], show=self.show,
                          share_axes=False,
                          labels=['a', 'b', 'c'])
        _assert_list(ax, out)
        return

    def test_list_of_series_with_labels(self):
        data = pd.Series(np.random.random(100))
        ax, out = boxplot([data, data, data], show=self.show,
                          share_axes=False,
                          labels=['a', 'b', 'c'])
        _assert_list(ax, out)
        return


def _assert_output(ax, out):
    assert isinstance(ax, plt.Axes), ax
    assert isinstance(out, dict)
    plt.close('all')
    return


def _assert_list(axes, outs):
    assert isinstance(axes, list)
    for out in outs:
        assert isinstance(out, dict)
    assert isinstance(outs, list)
    for ax in axes:
        assert isinstance(ax, plt.Axes)
    return


if __name__ == "__main__":
    unittest.main()
