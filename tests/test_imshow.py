
import unittest

import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easy_mpl import imshow


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

    def test_white_gridlines(self):
        data = np.random.random((4, 10))
        ax, im = imshow(data, cmap="YlGn",
                        xticklabels=[f"Feature {i}" for i in range(data.shape[1])],
                        white_grid=True, annotate=True, colorbar=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_df_object_type(self):
        data = pd.DataFrame(np.random.random((10, 10)), dtype='object')
        ax, im = imshow(data, white_grid=True, annotate=True, show=self.show)
        assert isinstance(ax, plt.Axes)
        return

    def test_textcolors_tuple(self):
        data = pd.DataFrame(np.random.random((10, 10)), dtype='object')
        ax, im = imshow(data, white_grid=True, annotate=True, show=self.show,
                        annotate_kws={'textcolors': ("black", "white")}
                        )
        assert isinstance(ax, plt.Axes)
        return

    def test_textcolors_array(self):
        data = pd.DataFrame(np.random.random((3, 3)), dtype='object')
        ax, im = imshow(data, cmap="YlGn",
       white_grid=True, annotate=True,
       annotate_kws={
              "textcolors": np.array([['black', 'black', 'black'],
                                      ['white', 'white', 'white'],
                                     ['green', 'green', 'green']])
       },
       show=self.show)
        assert isinstance(ax, plt.Axes)
        return



if __name__ == "__main__":
    unittest.main()
