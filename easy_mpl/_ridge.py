
__all__ = ["ridge"]

import random
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from .utils import kde
from .utils import is_dataframe
from .utils import make_cols_from_cmap
from .utils import despine_axes

# todo, single array with labels argument

# colormaps for ridge plot
RIDGE_CMAPS = [
    "afmhot", "afmhot_r", "Blues", "bone",
    "BrBG", "BuGn", "coolwarm", "cubehelix",
    "gist_earth", "GnBu", "Greens", "magma",
    "ocean", "Pastel1", "pink", "PuBu", "PuBuGn",
    "RdBu", "Spectral",
]


def ridge(
        data: Union[np.ndarray, List[np.ndarray]],
        bw_method="scott",
        cut: float = 0.5,
        color: Union[str, List[str], np.ndarray, List[np.ndarray]] = None,
        fill_kws: dict = None,
        line_width: Union[int, List[int], float, List[float]] = 1.0,
        line_color: Union[str, List[str]] = "black",
        plot_kws: dict = None,
        xlabel: str = None,
        title: str = None,
        figsize: tuple = None,
        hspace: float = -0.7,
        labels: Union[str, List[str]] = None,
        share_axes: bool = False,
        ax: plt.Axes = None,
        show=True,
) -> List[plt.Axes,]:
    """
    plots distribution of features/columns/arrays in data as ridge.

    Parameters
    ----------
        data : array, DataFrame
            array or list of arrays or pandas DataFrame/Series
        bw_method : str, optional
        cut : float (default=0.5)
        color : str, optional
            color to fill the ridges. It can be any valid matplotlib color or color
            name or cmap name or a list of colors for each ridge.
        fill_kws : dict, (default=None)
            keyword arguments that will go to :obj:`matplotlib.axes.Axes.fill_between`
        line_width : int (default=1.0)
            with of line of ridges.
        line_color : str (default="black")
            color or colors of lines of ridges.
        plot_kws : dict optional
            any keyword argumenets that will go to :obj:`matplotlib.axes.Axes.plot` during plot of kde line
        xlabel : str, optional
            xlabel for the figure
        title : str, optional
            title of the figure
        figsize : tuple, optional
            size of figure
        hspace : float, optional (default=-0.7)
            amount of distance between plots
        labels : list/str (default=None
            names for xticks
        share_axes : bool, optional (default=False)
            whether to draw all ridges on same axes or separate axes
        ax : plt.Axes, optional (default=None)
            matplotlib axes object :obj:`matplotlib.axes` on which to draw the ridges.
             If given all ridges will be drawn on this axes.
        show : bool, optional
            whether to show the plot or not

    Returns
    -------
    list
        a list of :obj:`matplotlib.axes`

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl import ridge
    >>> data_ = np.random.random((100, 3))
    >>> ridge(data_)
    ... # specifying colormap
    >>> ridge(data_, cmap="Blues")
    ... # using pandas DataFrame
    >>> import pandas as pd
    >>> ridge(pd.DataFrame(data_))

    See :ref:`sphx_glr_auto_examples_ridge.py` for more examples

    """

    if isinstance(data, np.ndarray):
        if len(data) == data.size:
            data = data.reshape(-1, 1)
        assert data.ndim == 2
        data = [data[:, i] for i in range(data.shape[1])]

    elif hasattr(data, 'name') and hasattr(data, 'values'):
        if labels is None:
            labels = [data.name]
        data = [data.values]
    elif is_dataframe(data):
        if labels is None:
            labels = data.columns.tolist()
        data = [data.values[:, i] for i in range(data.shape[1])]

    assert isinstance(data, list)

    if labels is None:
        labels =  [f"F{i}" for i in range(len(data))]

    n = len(data)

    if color is None:
        cmap = random.choice(RIDGE_CMAPS)
        # +2 because we want to ignore first and last in most cases
        colors = make_cols_from_cmap(cmap, n + 2)
    elif isinstance(color, str):
        if color in plt.colormaps():
            colors = make_cols_from_cmap(color, n + 2)
        else:
            colors = [color for _ in range(n)]
    elif isinstance(color, list):
        colors = color
    elif isinstance(color, np.ndarray):
        if len(color) == 3 and len(color) == color.size:
            colors = [color for _ in range(n)]
        else:
            colors = [color[:, i] for i in range(color.shape[1])]
    else:
        colors = color

    if plot_kws is None:
        plot_kws = dict()

    dist_maxes = {}
    xs = {}
    ys = {}
    for idx, col in enumerate(labels):
        ind, y = kde(data[idx], bw_method=bw_method, cut=cut)
        dist_maxes[col] = np.max(y)
        xs[col], ys[col] = ind, y

    ymaxes = dict(sorted(dist_maxes.items(), key=lambda item: item[1]))

    nrows = n
    if share_axes:
        nrows = 1

    if ax is None:
        fig, axes = plt.subplots(nrows, ncols=1,
                                 figsize=figsize or (10, 6),
                                 gridspec_kw={"hspace": hspace})
    else:
        share_axes = True
        axes = ax
        nrows = 1

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_objs = axes.tolist()

    if isinstance(line_color, str):
        line_color = [line_color for _ in range(len(ymaxes))]

    if isinstance(line_width, (int, float)):
        line_width = [line_width for _ in range(len(ymaxes))]

    for idx, col in enumerate(reversed(list(ymaxes.keys()))):

        ax_idx = idx
        if share_axes:
            ax_idx = 0

        if isinstance(colors[idx], str) and colors[idx] == "white":
            plot_kws['label'] = col

        ax = ax_objs[ax_idx]
        ax.plot(xs[col], ys[col],
                         color=line_color[idx],
                         linewidth=line_width[idx], **plot_kws)

        _fill_kws = {
            "alpha": 1.0,
            'where': np.array([True for _ in range(len(xs[col]))])
        }

        if fill_kws is None:
            fill_kws = dict()

        if isinstance(colors[idx], str) and colors[idx] == "white":
            pass
        else:
            fill_kws['label'] = col

        _fill_kws.update(fill_kws)

        ax.fill_between(xs[col], ys[col], color=colors[idx], **_fill_kws)

        # setting uniform y lims
        ax.set_ylim(0, max(ymaxes.values()))

        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        if idx == n - 1:
            if xlabel:
                ax.set_xlabel(xlabel)

            ax.tick_params(axis="x")
        else:
            if nrows > 1:
                ax.set_xticklabels([])
                ax.set_xticks([])

        # remove borders, axis ticks, and labels
        if nrows > 1:
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis="y")

        despine_axes(ax)

        if not share_axes and nrows>1:
            ax.text(xs[col][0],
                             0.2,
                             col,
                             ha="right")
        idx += 1

    if not share_axes:
        plt.subplots_adjust(hspace=hspace)

    if title:
        plt.suptitle(title)

    if show:
        if share_axes:
            plt.legend()

        plt.show()

    return ax_objs
