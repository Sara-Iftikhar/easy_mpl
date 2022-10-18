
import unittest

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
        return

    def test_df_1col(self):
        axes = violin_plot(df[cols[-1:]], show=self.show)
        return

    def test_np(self):
        axes = violin_plot(df[cols].values, show=self.show)
        return

    def test_np_1array(self):
        axes = violin_plot(df[cols[0]].values, show=self.show)
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


if __name__ == "__main__":
    unittest.main()