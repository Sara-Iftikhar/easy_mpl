
__all__ = ["hist"]

from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axis


def hist(
        x: Union[list, np.ndarray],
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
            array like, dimensions must not be greader than 1d
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

    if isinstance(x, np.ndarray):
        if len(x) == x.size:
            X = [x]
            names = [None]
        else:
            X = [x[:, i] for i in range(x.shape[1])]
            names = [f"{i}" for i in range(x.shape[1])]
    elif is_dataframe(x):
        X = []
        for col in x.columns:
            X.append(x[col].values)
        names = x.columns.tolist()
    elif is_series(x):
        X = x.values
        names = [x.name]
    elif isinstance(x, (list, tuple)) and not is_dataframe(x[0]):
        X = [x]
        names = [None]
    else:
        raise ValueError(f"unrecognized type of x {type(x)}")

    for x, name in zip(X, names):
        if name:
            hist_kws['label'] = name

        n, bins, patches = ax.hist(x, **hist_kws)

    process_axis(ax, grid=grid, **kwargs)

    if name:
        plt.legend()

    if show:
        plt.show()

    return ax

def is_dataframe(x):
    if all([hasattr(x, attr) for attr in ["columns", "values", "index"]]):
        return True
    return False


def is_series(x):
    if all([hasattr(x, attr) for attr in ["name", "index", "values"]]):
        return True
    return False
