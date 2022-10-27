
__all__ = ["boxplot"]

from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axis
from .utils import make_cols_from_cmap


def boxplot(
        data:Union[np.ndarray, List[np.ndarray]],
        line_color:Union[str, List[str]] = None,
        line_width = None,
        fill_color:Union[str, List[str]] = None,
        ax:plt.Axes = None,
        show:bool = True,
        ax_kws:dict = None,
        **box_kws,
)->Tuple[plt.Axes, dict]:
    """
    Draws the box and whiker plot

    parameters
    ----------
    data :
        array or list of arrays
    line_color :
        name of color/colors/cmap lines/boundaries of box
    line_width :
        width of the box lines
    fill_color :
        name of color/colors/cmap to fill the boxes
    ax : plt.Axes, optional (default=None)
        matploltib axes on which to draw the plot
    show : bool (default=show)
        whether to show the plot or not
    ax_kws : dict (default=None)
        keyword arguments of :func:`process_axis`
    **box_kws :
        any additional keyword argument for axes.boxplot_

    Returns
    -------
    tuple
        a tuple of two
            plt.Axes
            a dictionary which consists of boxes, medians, whiskers, fliers

    Examples
    ---------
    >>> from easy_mpl import boxplot
    >>> boxplot(np.random.random((100, 5)))
    we can also provide arrays of different lengths
    >>> boxplot([np.random.random(100), np.random.random(90)])
    the color can be given as either color name or colormap
    >>> boxplot(np.random.random((100, 3)), fill_color=['pink', 'lightblue', 'lightgreen'])
    >>> boxplot(np.random.random((100, 3)), fill_color="viridis")

    See :ref:`sphx_glr_auto_examples_boxplot.py` for more examples

    .. _axes.boxplot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html
    """

    if ax is None:
        ax = plt.gca()

    _box_kws = {
    }

    xticklabels = None
    if hasattr(data, "columns"):
        xticklabels = data.columns

    if box_kws is None:
        box_kws = dict()

    _box_kws.update(box_kws)

    box_out = ax.boxplot(data, **_box_kws)

    if isinstance(fill_color, str) or is_rgb(fill_color):
        if isinstance(fill_color, str) and fill_color in plt.colormaps():
            fill_color = make_cols_from_cmap(fill_color, len(box_out['boxes']))  # name of cmap
        else:
            fill_color = [fill_color for _ in range(len(box_out['boxes']))]   # name of color

    if isinstance(line_color, str) or is_rgb(line_color):
        if isinstance(line_color, str) and line_color in plt.colormaps():
            fill_color = make_cols_from_cmap(fill_color, len(box_out['boxes']))  # name of cmap
        else:
            line_color = [line_color for _ in range(len(box_out['boxes']))]

    if isinstance(line_width, (float, int)):
        line_width = [line_width for _ in range(len(box_out['boxes']))]

    for idx, patch in enumerate(box_out['boxes']):
        if hasattr(patch, 'set_facecolor'):
            if fill_color is not None:
                patch.set_facecolor(fill_color[idx])

        if hasattr(patch, 'set_color') and line_color is not None:
            patch.set_color(line_color[idx])

        if hasattr(patch, 'set_linewidth') and line_width is not None:
            patch.set_linewidth(line_width[idx])

    if xticklabels is not None:
        kws = dict()
        if len(xticklabels)>7:
            kws['rotation'] = 90
        ax.set_xticks(range(len(xticklabels) + 1))
        ax.set_xticklabels(xticklabels.insert(0, ''), **kws)

    if ax_kws:
        process_axis(ax, **ax_kws)

    if show:
        plt.show()

    return ax, box_out


def is_rgb(color)->bool:
    if isinstance(color, list) and len(color)==3 and isinstance(color[0], (int, float)):
        return True
    return False
