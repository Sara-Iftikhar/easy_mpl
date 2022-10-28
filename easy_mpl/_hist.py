
__all__ = ["hist"]

from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axis, is_dataframe, create_subplots


def hist(
        x: Union[list, np.ndarray],
        hist_kws: dict = None,
        share_axes:bool = True,
        grid: bool = True,
        ax: plt.Axes = None,
        subplots_kws:dict = None,
        show: bool = True,
        **kwargs
) -> plt.Axes:
    """
    one stop shop for histogram

    Parameters
    -----------
        x : list, array, optional
            array like, numpy ndarray or pandas DataFrame, or list of arrays
        hist_kws : dict, optional
            any keyword arguments for `axes.hist`_
        share_axes : bool (default=True)
            whether to draw all the histograms on one axes or not?
        grid : bool, optional
            whether to show the grid or not
        show : bool, optional
            whether to show the plot or not
        ax : plt.Axes, optional
            axes on which to draw the plot
        subplots_kws : dict
            kws which go to plt.subplots() such as figure size (width, height)
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

    See :ref:`sphx_glr_auto_examples_hist.py` for more examples

    .. _axes.hist:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    """


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

    elif isinstance(x, (list, tuple)) and isinstance(x[0], (list, tuple)):
        X = [x_ for x_ in x]
        names = [None]*len(X)

    elif isinstance(x, (list, tuple)) and not is_dataframe(x[0]):
        X = [x]
        names = [None]
    else:
        raise ValueError(f"unrecognized type of x {type(x)}")

    #nplots = len(names)
    if share_axes:
        nplots = 1
    else:
        nplots = len(names)

    if nplots == 1:
        share_axes = True

    if subplots_kws is None:
        subplots_kws = dict()

    f, axes = create_subplots(nplots, ax=ax, **subplots_kws)

    if isinstance(axes, np.ndarray):
        axes = axes.flat

    for idx, x, name in zip(range(len(names)), X, names):
        if name:
            hist_kws['label'] = name

        if share_axes:
            assert isinstance(axes, plt.Axes)
            ax = axes
        else:
            ax = axes[idx]

        n, bins, patches = ax.hist(x, **hist_kws)

        process_axis(ax, grid=grid, **kwargs)

        if name:
            ax.legend()

    if show:
        plt.show()

    return ax


def is_series(x):
    if all([hasattr(x, attr) for attr in ["name", "index", "values"]]):
        return True
    return False
