

import random
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from .utils import kde
from .utils import make_cols_from_cmap
from .utils import RIDGE_CMAPS


def ridge(
        data: Union[pd.DataFrame, np.ndarray],
        cmap: str = None,
        xlabel: str = None,
        title: str = None,
        figsize: tuple = None,
        show=True
) -> List[plt.Axes,]:
    """
    plots distribution of features/columns/arrays in data as ridge.

    Parameters
    ----------
        data : array, DataFrame
            2 dimensional array. It must be either numpy array or pandas dataframe

        cmap : str, optional
        xlabel : str, optional
        title : str, optional
        figsize : tuple, optional
            size of figure
        show : bool, optional

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

    if cmap is None:
        cmap = random.choice(RIDGE_CMAPS)

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        assert data.ndim == 2
        data = pd.DataFrame(data,
                            columns=[f"Feature_{i}" for i in range(data.shape[1])])
    elif isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    else:
        assert isinstance(data, pd.DataFrame)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col])

    # +2 because we want to ignore first and last in most cases
    colors = make_cols_from_cmap(cmap, data.shape[1] + 2)

    gs = grid_spec.GridSpec(len(data.columns), 1)
    fig = plt.figure(figsize=figsize or (16, 9))

    dist_maxes = {}
    xs = {}
    ys = {}
    for col in data.columns:
        ind, y = kde(data[col].dropna())
        dist_maxes[col] = np.max(y)
        xs[col], ys[col] = ind, y

    ymaxes = dict(sorted(dist_maxes.items(), key=lambda item: item[1]))

    ax_objs = []

    for idx, col in enumerate(reversed(list(ymaxes.keys()))):

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[idx:idx + 1, 0:]))

        # plotting the distribution
        # todo, remove pandas from here, we already calculated kde
        plot_ = data[col].plot.kde(ax=ax_objs[-1])

        _x = plot_.get_children()[0]._x
        _y = plot_.get_children()[0]._y
        ax_objs[-1].fill_between(_x, _y, alpha=1, color=colors[idx + 1])

        # setting uniform y lims
        ax_objs[-1].set_ylim(0, max(ymaxes.values()))

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        if idx == data.shape[-1] - 1:
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

    # plt.tight_layout()

    if show:
        plt.show()

    return ax_objs
