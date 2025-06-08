
__all__ = ["boxplot"]

import warnings
from itertools import zip_longest
from typing import Union, List, Tuple

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from .utils import is_rgb
from .utils import is_series
from .utils import is_dataframe
from .utils import process_axes
from .utils import create_subplots
from .utils import make_cols_from_cmap


def boxplot(
        data:Union[np.ndarray, List[np.ndarray], List[list], list, "Series", "DataFrame"],
        line_color:Union[str, List[str]] = None,
        line_width = None,
        fill_color:Union[str, List[str], List[float], np.ndarray] = None,
        labels:Union[str, List[str]] = None,
        share_axes:bool = True,
        figsize:tuple = None,
        ax:plt.Axes = None,
        ax_kws:dict = None,
        show:bool = True,
        **box_kws,
)->Tuple[Union[plt.Axes, List[plt.Axes]], Union[List[dict], dict]]:
    """
    Draws the box and whiker plot

    parameters
    ----------
    data :
        array like (list, numpy array, pandas dataframe/series) or list of
        array likes. If list of array likes, the length of arrays in the list
        can be different.
    line_color :
        name of color/colors/cmap for lines/boundaries of box, whisker, cap, median
        and mean line
    line_width :
        width of the box lines.
    fill_color :
        name of color/colors/cmap to fill the boxes. It can be any valid
        matplotlib color or cmap.
    labels : str/list (default=None)
        used for ticklabels of x-axes
    share_axes : bool (default=True)
        whether to draw all the histograms on one axes or not
    figsize : tuple (default=None)
        figure size as tuple (width, height)
    ax : plt.Axes, optional (default=None)
        matploltib axes on which to draw the plot
    ax_kws : dict (default=None)
        keyword arguments of :py:func:`easy_mpl.utils.process_axes`
    show : bool (default=show)
        whether to show the plot or not
    **box_kws :
        any additional keyword argument for :obj:`matplotlib.axes.Axes.boxplot`

    Returns
    -------
    tuple
        a tuple of two
            - plt.Axes or list of :obj:`matplotlib.axes`
            - a dictionary or list of dictionaries which consists of boxes,
              medians, whiskers, fliers

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

    """

    if ax is None:
        ax = plt.gca()

    _box_kws = {}

    data, labels = _unpack_data(data, labels, share_axes)

    if share_axes:
        nplots = 1
    else:
        nplots = len(labels)

    if box_kws is None:
        box_kws = dict()

    _box_kws.update(box_kws)

    f, axes = create_subplots(nplots, ax=ax, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    elif isinstance(axes, plt.Axes):
        axes = [axes]

    nboxes = len(labels)
    if len(labels)==1:
        nboxes = len(labels[0])
    fill_colors = _unpack_colors(fill_color, nboxes, share_axes)
    line_colors = _unpack_colors(line_color, nboxes, share_axes)
    line_widths = _unpack_linewidth(line_width, nboxes, share_axes)

    box_outs = []
    for (idx, name), x, ax in zip(enumerate(labels), data, axes):

        # in version 3.2.. giving DataFrame to ax.boxplot makes boxes for each row
        # in version 3.3.. giving DataFrame to ax.boxplot tries to make boxp for first row (columns)
        if is_dataframe(x) and matplotlib.__version__ <= "3.3.0":
            x = x.values

        box_out = ax.boxplot(x, **_box_kws)
        box_outs.append(box_out)

        _set_box_props(fill_colors[idx], line_colors[idx],
                       line_widths[idx], box_out)

        _set_ticklabels(ax, share_axes, name, _box_kws)

        if ax_kws:
            process_axes(ax, **ax_kws)

    if show:
        plt.show()

    if len(box_outs)==1:
        box_outs = box_outs[0]

    if len(axes)==1:
        axes = axes[0]

    return axes, box_outs


def _set_ticklabels(ax, share_axes, name, box_kws):

    if name is not None:
        kws = dict()

        if share_axes:
            if box_kws.get('vert', True):
                _maybe_set_ticks(ax, name)
            else:
                _maybe_set_ticks(ax, name, "y")
        else:
            if isinstance(name, (str, int)):
                if box_kws.get('vert', True):
                    ax.set_xticklabels([name])
                else:
                    ax.set_yticklabels([name], rotation=90, va='center')
            elif isinstance(name, list):
                if box_kws.get('vert', True):
                    ax.set_xticklabels(name)
                else:
                    ax.set_yticklabels(name, rotation=90, va='center')

        if share_axes and len(name) > 7:
            kws['rotation'] = 90
            ax.xaxis.set_tick_params(rotation=90)

    return


def _maybe_set_ticks(axes:plt.Axes, ticklabels, which="x"):
    ticks = getattr(axes, f"get_{which}ticks")()

    if len(ticklabels) == len(ticks):
        getattr(axes, f"set_{which}ticklabels")(ticklabels)
    else:
        warnings.warn(f"""
{which}ticks ({len(ticks)}) and {which}ticklabels ({len(ticklabels)}) dont match""")
    return


def _unpack_linewidth(line_width, nboxes, share_axes):
    if isinstance(line_width, (float, int)):
        line_widths = [[line_width] for _ in range(nboxes)]
    elif line_width is None:
        line_widths = [[None] for _ in range(nboxes)]
    else:
        raise ValueError

    if share_axes:
       line_widths = [[line_width[0] for line_width in line_widths]]

    return line_widths


def _unpack_colors(color, nboxes, share_axes)->list:
    if isinstance(color, str):
        if color in plt.colormaps():
            colors = make_cols_from_cmap(color, nboxes)
            colors = [[color] for color in colors]
        else:
            colors = [[color] for _ in range(nboxes)]
    elif is_rgb(color):
        colors = [[color] for _ in range(nboxes)]
    elif hasattr(color, '__len__'):
        assert len(color) == nboxes, f"{len(color)} colors for {nboxes} boxes?"
        colors = [[clr] for clr in color]
    elif color is None:
        colors = [[None] for _ in range(nboxes)]
    else:
        raise ValueError(f"{color} is not recognized as valid color")

    if share_axes:
       colors = [[color[0] for color in colors]]

    return colors


def _set_box_props(fill_color:list,
                   line_color:list,
                   line_width:list,
                   box_out):

    whiskers = box_out['whiskers']
    if len(whiskers)>0:
        whiskers = np.array(whiskers).reshape(len(line_color), -1)

    boxes = box_out['boxes']
    if len(boxes)>0:
        boxes = np.array(boxes).reshape(len(line_color), -1)

    caps = box_out['caps']
    if len(caps)>0:
        caps = np.array(caps).reshape(len(line_color), -1)

    medians = box_out['medians']
    if len(medians)>0:
        medians = np.array(medians).reshape(len(line_color), -1)

    means = box_out['means']
    if len(means)>0:
        means = np.array(means).reshape(len(line_color), -1)

    fliers = box_out['fliers']
    if len(fliers)>0:
        fliers = np.array(fliers).reshape(len(line_color), -1)

    for idx, (patch, whisker, cap, median, mean, flier) in enumerate(zip_longest(
            boxes, whiskers, caps, medians, means, fliers)):

        if fill_color[idx] is not None:
            if isinstance(patch[0], matplotlib.lines.Line2D):
                plt.setp(patch, markerfacecolor=fill_color[idx])
            elif isinstance(patch[0], matplotlib.patches.PathPatch):
                plt.setp(patch, facecolor=fill_color[idx])

        if line_color[idx] is not None:
            #plt.setp(patch, color=line_color[idx])
            plt.setp(whisker, color=line_color[idx])
            plt.setp(cap, color=line_color[idx])
            plt.setp(median, color=line_color[idx])

        if line_width[idx] is not None:
            plt.setp(patch, linewidth=line_width[idx])
            plt.setp(whisker, linewidth=line_width[idx])
            plt.setp(cap, linewidth=line_width[idx])
            plt.setp(median, linewidth=line_width[idx])
    return


def _unpack_data(x, labels, share_axes:bool)->Tuple[list, list]:

    if isinstance(x, np.ndarray):
        if len(x) == x.size:
            X = [x]
            names = [[None]]
        else:
            if share_axes:
                X = [x]
                names = [[f"{i}" for i in range(x.shape[1])]]
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
        X = [x.values]
        names = [[x.name]]

    elif isinstance(x, (list, tuple)) and isinstance(x[0], (list, tuple, np.ndarray)):

        assert all([len(array) == np.array(array).size for array in x]), f"""
        All arrays must be one dimensional."""

        X = [np.array(x_).reshape(-1,) for x_ in x]
        names = [None] * len(X)
        if share_axes:
            X = [X]
            names = [names]

    elif isinstance(x, (list, tuple)) and is_series(x[0]):  # list of series
        if share_axes:
            X = [x]
        else:
            X = x
        names = [x_.name for x_ in x]

    elif isinstance(x, (list, tuple)) and not is_dataframe(x[0]):
        X = [x]
        names = [None]
    else:
        raise ValueError(f"unrecognized type of x {type(x)}")

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if share_axes:
            labels = [labels]
        #assert len(labels) == len(names), f"{len(names)} does not match data"
        names = labels

    return X, names
