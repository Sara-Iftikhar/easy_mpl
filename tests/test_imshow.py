
import unittest

import os
import site

import matplotlib.pyplot as plt

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import matplotlib
import numpy as np
import pandas as pd

from easy_mpl import imshow


class TestImshow(unittest.TestCase):

    show = False
    x = np.random.random((20, 20))

    def test_imshow(self):
        img = imshow(np.random.random((10, 10)), colorbar=True,
                     ax_kws=dict(title="vanilla"),
                       show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_figsize(self):
        img = imshow(np.random.random((10, 10)), colorbar=True,
            ax_kws=dict(title="figsize", figsize=(10, 10)),
                       show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_imshow_witout_cb(self):
        img= imshow(np.random.random((10, 10)), colorbar=False,
                    ax_kws=dict(title="without_colorbar"),
                        show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_annotate(self):
        img= imshow(np.random.random((10, 10)), colorbar=False,
                    ax_kws=dict(title="annotate"),
                        annotate=True, show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_df(self):
        df = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
        img = imshow(df, colorbar=True, show=self.show, ax_kws=dict(title="df"))
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_with_nan_vals(self):
        x = np.random.random((10, 10))
        x.ravel()[np.random.choice(x.size, 5, replace=False)] = np.nan
        img = imshow(x, colorbar=True, show=self.show, ax_kws=dict(title="with_nan_vals"))
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_white_gridlines(self):
        data = np.random.random((4, 10))
        im = imshow(data, cmap="YlGn",
                        xticklabels=[f"Feature {i}" for i in range(data.shape[1])],
                        grid_params={}, annotate=True, colorbar=True, show=self.show)
        assert isinstance(im, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_df_object_type(self):
        data = pd.DataFrame(np.random.random((10, 10)), dtype='object')
        img = imshow(data, grid_params={}, annotate=True, show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_textcolors_tuple(self):
        data = pd.DataFrame(np.random.random((10, 10)), dtype='object')
        img = imshow(data, grid_params={}, annotate=True, show=self.show,
                        annotate_kws={'textcolors': ("black", "white")}
                        )
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_textcolors_array(self):
        data = pd.DataFrame(np.random.random((3, 3)), dtype='object')
        img = imshow(data, cmap="YlGn",
        annotate=True,
        annotate_kws={
              "textcolors": np.array([['black', 'black', 'black'],
                                      ['white', 'white', 'white'],
                                     ['green', 'green', 'green']])
        },
        show=self.show)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return

    def test_mask_true(self):
        imshow(self.x, mask=True, show=self.show)
        return

    def test_mask_upper(self):
        imshow(self.x, mask=True, show=self.show)
        return

    def test_mask_lower(self):
        imshow(self.x, mask=True, show=self.show)
        return



if __name__ == "__main__":
    unittest.main()
