
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl._violin import violin_plot
from easy_mpl.utils import _rescale


f = "https://raw.githubusercontent.com/AtrCheema/AI4Water/master/ai4water/datasets/arg_busan.csv"
df = pd.read_csv(f, index_col='index')
cols = ['air_temp_c', 'wat_temp_c', 'sal_psu', 'tide_cm', 'rel_hum', 'pcp12_mm']
for col in df.columns:
    df[col] = _rescale(df[col].values)


class TestViolin(unittest.TestCase):
    show = False

    def test_df(self):
        axes = violin_plot(df[cols], show=self.show)
        axes = violin_plot(df[cols], show=self.show, index_method="kde")
        return

    def test_cut(self):
        axes = violin_plot(df[cols], show=self.show, cut=0.1)
        axes = violin_plot(df[cols], show=self.show, cut=0.2)
        axes = violin_plot(df[cols], show=self.show, cut=0.5)
        return

    def test_df_1col(self):
        axes = violin_plot(df[cols[-1:]], show=self.show)
        axes = violin_plot(df[cols[-1:]], show=self.show, index_method="kde")
        return

    def test_np(self):
        axes = violin_plot(df[cols].values, show=self.show)
        axes = violin_plot(df[cols].values, show=self.show, index_method="kde")
        return

    def test_np_1array(self):
        Y = np.random.gamma(20, 10, 100)
        axes = violin_plot(Y, show=self.show)
        axes = violin_plot(Y, show=self.show, index_method="kde")
        return

    def test_swarm(self):
        axes = violin_plot(df[cols], show=self.show, show_datapoints=True)
        return

    def test_show_boxplot(self):
        axes = violin_plot(df[cols], show=self.show, show_boxplot=True)
        return

    def test_scatter_kws(self):
        axes = violin_plot(df[cols], show=self.show,
                           scatter_kws={"s": 10, 'alpha': 0.4},
                           )
        return

    def test_fillcolors_as_rgb(self):
        axes = violin_plot(df[cols], show=self.show,
                           fill_colors=[np.array([253, 160, 231]) / 255,
                                        np.array([102, 217, 191]) / 255,
                                        np.array([251, 173, 167]) / 255,
                                        np.array([253, 160, 231]) / 255,
                                        np.array([102, 217, 191]) / 255,
                                        np.array([251, 173, 167]) / 255],

                           )

        return

    def test_return_axes(self):
        axes = violin_plot(df[cols], show=self.show)
        axes.set_xticks(range(3))
        axes.set_xticklabels(["Pb", "Cr", "Hg"], size=15, ha="center", ma="center")
        axes.set_facecolor("#fbf9f4")
        if self.show:
            plt.show()
        return

    def test_cut_as_list(self):
        axes = violin_plot(df[cols], show=self.show, cut=[0.1,0.2,0.3,0.4,0.5,0.6])
        axes = violin_plot(df[cols], show=self.show, index_method="kde",
                           cut=[0.1,0.2,0.3,0.4,0.5,0.6])
        return

    def test_cut_tuple(self):
        axes = violin_plot(df[cols], show=self.show, cut=(0.1, 0.2))
        axes = violin_plot(df[cols], show=self.show, index_method="kde", cut=(0.1, 0.2))
        return

    def test_cut_as_list_of_tuple(self):
        cut = [(0.1, 0.2), (0.11, 0.21), (0.12, 0.22), (0.13, 0.23),
               (0.14, 0.24), (0.1, 0.2)]
        axes = violin_plot(df[cols], show=self.show, cut=cut)
        axes = violin_plot(df[cols], show=self.show, index_method="kde", cut=cut)
        return

    def test_max_dots_as_list(self):
        axes = violin_plot(df[cols], show=self.show, max_dots=[50, 60, 70, 50, 60, 70])
        axes = violin_plot(df[cols], show=self.show, index_method="kde",
                           max_dots=[50, 60, 70, 50, 60, 70])
        return

    def test_labels(self):
        data = np.random.random((100, 3))
        violin_plot(data, show=self.show, labels=['a', 'b', 'c'])
        return

if __name__ == "__main__":
    unittest.main()