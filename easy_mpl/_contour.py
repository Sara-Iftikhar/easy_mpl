
__all__ = ["contour"]

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axes


def contour(
        x,
        y,
        z,
        fill_between: bool = False,
        show_points: bool = False,
        colorbar: bool = True,
        label_contours: bool = False,
        fill_between_kws: dict = None,
        show_points_kws: dict = None,
        label_contour_kws: dict = None,
        ax: plt.Axes = None,
        show: bool = True,
        ax_kws:dict = None,
        **kwargs
):
    """A contour plot of irregularly spaced data coordinates.

    Parameters
    ---------
        x : array, list
            a 1d array defining grid positions along x-axis
        y : array, list
            a 1d array defining grid positions along y-axis
        z : array, list
            values on grids/points defined by x and y
        fill_between : bool, optional
            whether to fill the space between colors or not
        show_points : bool, optional
            whether to show the
        label_contours : bool, optional
            whether to label the contour lines or not
        colorbar : bool, optional
            whether to show the colorbar or not. Only valid if `fill_between` is
            True
        fill_between_kws : dict, optional
            any keword argument for `axes.tricontourf`_ for example ``levels`` or
            ``cmap``.
        show_points_kws : dict, optional
            any keword argument for `axes.plot`_
        label_contour_kws : dict, optional
            any keword argument for `axes.clabel`_
        ax : plt.Axes, optional
            matplotlib axes to work on. If not given, current active axes will
            be used.
        ax_kws : dict optional
            any keyword arguments for :py:func:`easy_mpl.utils.process_axes`.
        show : bool, optional
            whether to show the plot or not
        **kwargs :
            any keword argument for `axes.tricontour`_

    Returns
    -------
    plt.Axes
        a matplotliblib Axes

    Examples
    --------
        >>> from easy_mpl import contour
        >>> import numpy as np
        >>> _x = np.random.uniform(-2, 2, 200)
        >>> _y = np.random.uniform(-2, 2, 200)
        >>> _z = _x * np.exp(-_x**2 - _y**2)
        >>> contour(_x, _y, _z, fill_between=True, show_points=True)
        ... # show contour labels
        >>> contour(_x, _y, _z, label_contours=True, show_points=True)

    See :ref:`sphx_glr_auto_examples_contour.py` for more examples

    Note
    ----
    The length of x and y should be same. The actual grid is created using axes.tricontour
    and axes.tricontourf functions.

    .. _axes.tricontour:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontour.html

    .. _axes.tricontourf:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontourf.html

    .. _axes.plot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

    .. _axes.clabel:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.clabel.html
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert len(x) == len(y) == len(z)

    if ax_kws is None:
        ax_kws = dict()

    if ax is None:
        ax = plt.gca()
        if 'figsize' in ax_kws:
            figsize = ax_kws.pop('figsize')
            ax.figure.set_size_inches(figsize)

    _kws = {"levels": 14, "linewidth": 0.5, "colors": "k"}
    if kwargs:
        _kws.update(kwargs)
    CS = ax.tricontour(x, y, z, **_kws)

    if fill_between:
        kws = fill_between_kws or {'levels': 14, 'cmap': 'RdBu_r'}
        CS = ax.tricontourf(x, y, z, **kws)

        if colorbar:
            fig: plt.Figure = plt.gcf()
            fig.colorbar(CS, ax=ax)

    if show_points:
        kws = show_points_kws or {'color': 'k', 'marker': 'o', 'ms': 3, "linestyle": ""}
        ax.plot(x, y, **kws)

    if ax_kws:
        process_axes(ax, **ax_kws)

    if label_contours:
        kws = label_contour_kws or {"colors": "black"}
        ax.clabel(CS, **kws)

    if show:
        plt.show()

    return ax
