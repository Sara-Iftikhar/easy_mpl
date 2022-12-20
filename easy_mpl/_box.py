
__all__ = ["boxplot"]

from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axis
from .utils import create_subplots
from .utils import make_cols_from_cmap
from .utils import is_dataframe
from .utils import is_series


def boxplot(
        data:Union[np.ndarray, List[np.ndarray]],
        line_color:Union[str, List[str]] = None,
        line_width = None,
        fill_color:Union[str, List[str]] = None,
        labels:Union[str, List[str]] = None,
        ax:plt.Axes = None,
        show:bool = True,
        ax_kws:dict = None,
        share_axes:bool = True,
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
    labels : str/list (default=None)
        used for ticklabels of x-axes
    ax : plt.Axes, optional (default=None)
        matploltib axes on which to draw the plot
    show : bool (default=show)
        whether to show the plot or not
    ax_kws : dict (default=None)
        keyword arguments of :func:`process_axis`
    share_axes : bool (default=True)
        whether to draw all the histograms on one axes or not?
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

    data, labels = _unpack_data(data, labels, share_axes)

    if share_axes:
        nplots = 1
    else:
        nplots = len(labels)

    if box_kws is None:
        box_kws = dict()

    _box_kws.update(box_kws)

    f, axes = create_subplots(nplots, ax=ax)

    if isinstance(axes, np.ndarray):
        axes = axes.flat
    elif isinstance(axes, plt.Axes):
        axes = [axes]

    box_outs = []
    for (idx, name), x, ax in zip(enumerate(labels), data, axes):
        box_out = ax.boxplot(x, **_box_kws)
        box_outs.append(box_out)

        _set_box_props(box_out, fill_color, line_color, line_width)

        if name is not None:
            kws = dict()

            if not share_axes:
                if isinstance(name, (str, int)):
                    ax.set_xticklabels([name])
                elif isinstance(name, list):
                    ax.set_xticklabels(name)

            if share_axes and len(name)>7:
                kws['rotation'] = 90
                ax.xaxis.set_tick_params(rotation=90)


        if ax_kws:
            process_axis(ax, **ax_kws)

    if show:
        plt.show()

    if len(box_outs)==1:
        box_outs = box_outs[0]

    if len(axes)==1:
        axes = axes[0]

    return axes, box_outs


def is_rgb(color)->bool:
    if isinstance(color, list) and len(color)==3 and isinstance(color[0], (int, float)):
        return True
    return False


def _set_box_props(box_out, fill_color, line_color, line_width):
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
    return


def _unpack_data(x, labels, share_axes:bool)->Tuple[list, list]:

    if isinstance(x, np.ndarray):
        if len(x) == x.size:
            X = [x]
            names = [None]
        else:
            X = [x[:, i] for i in range(x.shape[1])]
            names = [f"{i}" for i in range(x.shape[1])]

    elif is_dataframe(x):
        if share_axes:
            names = [x.columns.tolist()]
            X = [x]
        else:
            X = []
            for col in x.columns:
                X.append(x[col].values)
            names = x.columns.tolist()

    elif is_series(x):
        X = x.values
        names = [x.name]

    elif isinstance(x, (list, tuple)) and isinstance(x[0], (list, tuple, np.ndarray)):
        X = [x_ for x_ in x]
        names = [None]*len(X)

    elif isinstance(x, (list, tuple)) and not is_dataframe(x[0]):
        X = [x]
        names = [None]
    else:
        raise ValueError(f"unrecognized type of x {type(x)}")

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        assert len(labels) == len(names), f"{len(names)} does not match data"
        names = labels

    return X, names
