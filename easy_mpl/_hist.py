
__all__ = ["hist"]

from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import process_axis


def hist(
        x: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        hist_kws: dict = None,
        grid: bool = True,
        ax: plt.Axes = None,
        show: bool = True,
        **kwargs
) -> plt.Axes:
    """
    one stop shop for histogram

    Parameters
    -----------
        x : list, array, optional
            array like, must not be greader than 1d
        grid : bool, optional
            whether to show the grid or not
        show : bool, optional
            whether to show the plot or not
        ax : plt.Axes, optional
            axes on which to draw the plot
        hist_kws : dict, optional
            any keyword arguments for `axes.hist`_
        **kwargs : optional
            any keyword arguments for axes manipulation such as title, xlable, ylable etc

    matplotlib.pyplot.Axes
        matplotlib Axes on which the histogram is drawn. If ``show`` is False, this axes
        can be used for further processing

    Example
    --------
        >>> from easy_mpl import hist
        >>> import numpy as np
        >>> hist(np.random.random((10, 1)))

    .. _axes.hist:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    """

    if not ax:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    hist_kws = hist_kws or {}
    n, bins, patches = ax.hist(x, **hist_kws)

    process_axis(ax, grid=grid, **kwargs)

    if show:
        plt.show()

    return ax

