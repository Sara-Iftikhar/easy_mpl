
__all__ = ["lollipop_plot"]

import numpy as np
import matplotlib.pyplot as plt

from .utils import is_rgb
from .utils import to_1d_array, process_axes


def lollipop_plot(
        y,
        x=None,
        orientation: str = "vertical",
        sort: bool = False,
        line_style: str = '-',
        line_color: str = 'cyan',
        line_width: int = 1,
        line_kws: dict = None,
        marker_style: str = 'o',
        marker_color: str = 'teal',
        marker_size: int = 30,
        marker_kws: dict = None,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        show: bool = True,
        **kwargs
) -> plt.Axes:
    """
    Plot a lollipop plot.

    Parameters
    ----------
    y : array_like, shape (n,), optional
        The y-coordinates of the data points.
    x : array_like, shape (n,)
        The x-coordinates of the data points.
    orientation : str, optional
        The orientation of the lollipops. Either "vertical" or "horizontal".
    sort : bool, optional
        Whether to sort the data points by their values or not.
        Only valid if `x` is not specified.
    line_style : str, optional
        The line style of the data points.
    line_color : str, optional
        The line color of the data points.
    line_width : float, optional
        The line width of the data points.
    line_kws : dict, optional
        The keyword arguments for the line. These arguments are passed to
        `matplotlib.axes.Axes.plot`_.
    marker_style : str, optional
        The marker style of the data points.
    marker_color : str, optional
        The marker color of the data points.
    marker_size : float, optional
        The marker size of the data points.
    marker_kws : dict, optional
        The keyword arguments for the marker. These arguments are passed to
        `matplotlib.axes.Axes.scatter`_.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not given, current available axes will be used.
    ax_kws : dict, optional
        any keyword arguments for :py:func:`easy_mpl.utils.process_axes`.
    show : bool, optional (default=True)
        whether to show the plot or not
    **kwargs : optional
        Additional keyword arguments to pass to the process_axis function.

    Returns
    -------
    plt.Axes :
        The axes on which the plot was drawn.

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl import lollipop_plot
    >>> y = np.random.randint(0, 10, size=10)
    ... # vanilla lollipop plot
    >>> lollipop_plot(y, title="vanilla")
    ... # use both x and y
    >>> lollipop_plot(y, np.linspace(0, 100, len(y)), title="with x and y")
    ... # use custom line style
    >>> lollipop_plot(y, line_style='--', title="with custom linestyle")
    ... # use custom marker style
    >>> lollipop_plot(y, marker_style='D', title="with custom marker style")
    ... # sort the data points before plotting
    >>> lollipop_plot(y, sort=True, title="sort")
    ... # horzontal orientation of lollipops
    >>> y = np.random.randint(0, 20, size=10)
    >>> lollipop_plot(y, orientation="horizontal", title="horizontal")

    See :ref:`sphx_glr_auto_examples_lollipop_plot.py` for more examples

    .. _matplotlib.axes.Axes.plot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

    .. _matplotlib.axes.Axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    """
    if ax_kws is None:
        ax_kws = dict()

    if ax is None:
        ax = plt.gca()
        if 'figsize' in ax_kws:
            figsize = ax_kws.pop('figsize')
            ax.figure.set_size_inches(figsize)

    y = to_1d_array(y)

    if sort:
        idx = np.argsort(y)
        y = y[idx]
        assert x is None

        if isinstance(line_color, (list, np.ndarray, tuple)):
            if is_rgb(line_color[0]) or isinstance(line_color[0], str):
                line_color = np.array(line_color)[idx]

    if x is None:
        x = np.arange(len(y))

    x = to_1d_array(x)

    marker_kws = marker_kws or {}
    line_kws = line_kws or {}

    if orientation == "vertical":
        _lollipop_vertical(ax, x, y, line_style, line_color, line_width, line_kws,
                           marker_style, marker_color, marker_size, marker_kws)
    else:
        _lollipop_horizontal(ax, x, y, line_style, line_color, line_width, line_kws,
                             marker_style, marker_color, marker_size, marker_kws)

    if ax_kws and kwargs:
        process_axes(ax=ax, **ax_kws, **kwargs)

    if show:
        plt.show()

    return ax


def _lollipop_vertical(ax, x, y, line_style, line_color, line_width, line_kws,
                       marker_style, marker_color, marker_size, marker_kws):

    if isinstance(marker_color, str) and marker_color in plt.colormaps():
        marker_kws['cmap'] = marker_color
    else:
        marker_kws['color'] = marker_color

    ax.scatter(x, y, marker=marker_style,
               s=marker_size, **marker_kws)

    if line_color in plt.colormaps():
        line_kws['cmap'] = line_color
    else:
        line_kws['color'] = line_color

    ax.vlines(x, np.zeros(len(x)), y,
              linestyle=line_style, linewidth=line_width, **line_kws)
    return ax


def _lollipop_horizontal(ax, x, y, line_style, line_color, line_width, line_kws,
                         marker_style, marker_color, marker_size, marker_kws):
    ax.scatter(y, x, marker=marker_style, color=marker_color,
               s=marker_size, **marker_kws)
    ax.hlines(x, np.zeros(len(y)), y, color=line_color,
              linestyle=line_style, linewidth=line_width, **line_kws)
    return ax
