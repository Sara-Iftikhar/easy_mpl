
__all__ = ["ridge"]

import random
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from .utils import kde
from .utils import make_cols_from_cmap


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
        color: Union[str, List[str], np.ndarray, List[np.ndarray]] = None,
        fill_kws: dict = None,
        line_width:Union[int, List[int]] = 1.0,
        line_color:Union[str, List[str]] = "black",
        xlabel: str = None,
        title: str = None,
        figsize: tuple = None,
        show=True,
) -> List[plt.Axes,]:
    """
    plots distribution of features/columns/arrays in data as ridge.

    Parameters
    ----------
        data : array, DataFrame
            array or list of arrays or pandas DataFrame/Series
        color : str, optional
            color to fill the ridges. It can be any valid matplotlib color or color
             name or cmap name or a list of colors for each ridge.
        fill_kws : dict, (default=None)
            keyword arguments that will go to axes.fill_between
        line_width : int (default=1.0)
            with of line of ridges.
        line_color : str (default="black")
            color or colors of lines of ridges.
        xlabel : str, optional
        title : str, optional
        figsize : tuple, optional
            size of figure
        show : bool, optional
            whether to show the plot or not

    Returns
    -------
        list
            a list of plt.Axes

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

    """

    names = None
    if isinstance(data, np.ndarray):
        if len(data) == data.size:
            data = data.reshape(-1, 1)
        assert data.ndim == 2
        data = [data[:, i] for i in range(data.shape[1])]

    elif isinstance(data, pd.Series):
        names = [data.name]
        data = [data]
    elif hasattr(data, "columns") and hasattr(data, "values"):
        names = data.columns.tolist()
        data = [data.values[:, i] for i in range(data.shape[1])]

    assert isinstance(data, list)

    if names is None:
        names =  [f"F{i}" for i in range(len(data))]

    n = len(data)

    if color is None:
        cmap = random.choice(RIDGE_CMAPS)
        # +2 because we want to ignore first and last in most cases
        colors = make_cols_from_cmap(cmap, n + 2)
    elif isinstance(color, str):
        if color in plt.colormaps():
            colors = make_cols_from_cmap(color, n + 2)
        else:
            colors = [None] + [color for _ in range(n)]
    elif isinstance(color, list):
        colors = [None] + color
    elif isinstance(color, np.ndarray):
        if len(color)==3 and len(color) == color.size:
            colors = ["None"] + [color for _ in range(n)]
        else:
            colors = ["noen"] + [color[:, i] for i in range(color.shape[1])]
    else:
        colors = color

    gs = grid_spec.GridSpec(n, 1)
    fig = plt.figure(figsize=figsize or (16, 9))

    dist_maxes = {}
    xs = {}
    ys = {}
    for idx, col in enumerate(names):
        ind, y = kde(pd.Series(data[idx], name=col, dtype=np.float32).dropna())
        dist_maxes[col] = np.max(y)
        xs[col], ys[col] = ind, y

    ymaxes = dict(sorted(dist_maxes.items(), key=lambda item: item[1]))

    ax_objs = []

    if isinstance(line_color, str):
        line_color = [line_color for _ in range(len(ymaxes))]

    if isinstance(line_width, (int, float)):
        line_width = [line_width for _ in range(len(ymaxes))]

    for idx, col in enumerate(reversed(list(ymaxes.keys()))):

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[idx:idx + 1, 0:]))

        # plotting the distribution
        # todo, remove pandas from here, we already calculated kde
        _df = pd.Series(data[idx], name=col)
        plot_ = _df.plot.kde(ax=ax_objs[-1],
                                   color=line_color[idx],
                                   linewidth=line_width[idx])

        _x = plot_.get_children()[0]._x
        _y = plot_.get_children()[0]._y

        _fill_kws = {
            "alpha": 1.0
        }

        if fill_kws is None:
            fill_kws = dict()

        _fill_kws.update(fill_kws)

        ax_objs[-1].fill_between(_x, _y, color=colors[idx + 1], **_fill_kws)

        # setting uniform y lims
        ax_objs[-1].set_ylim(0, max(ymaxes.values()))

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        if idx == n - 1:
            if xlabel:
                ax_objs[-1].set_xlabel(xlabel, fontsize=26, fontweight="bold")

            ax_objs[-1].tick_params(axis="x", labelsize=20)
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xticks([])

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_ylabel('')
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_yticks([])

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(_x[0],
                         0.2,
                         col,
                         fontsize=20,
                         ha="right")
        idx += 1

    gs.update(hspace=-0.7)

    if title:
        plt.suptitle(title, fontsize=25)

    if show:
        plt.show()

    return ax_objs
