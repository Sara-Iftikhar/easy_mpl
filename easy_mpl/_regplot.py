
__all__ = ["regplot"]

import random
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import to_1d_array, _regplot
from .utils import regplot_combs


def regplot(
        x: Union[np.ndarray, pd.DataFrame, pd.Series, list],
        y: Union[np.ndarray, pd.DataFrame, pd.Series, list],
        title: str = None,
        annotation_key: str = None,
        annotation_val: float = None,
        line_color=None,
        marker_color=None,
        fill_color=None,
        marker_size: int = 20,
        ci: Union[int, None] = 95,
        figsize: tuple = None,
        xlabel: str = 'Observed',
        ylabel: str = 'Predicted',
        show: bool = True,
        ax: plt.Axes = None
) -> plt.Axes:
    """
    Regpression plot with regression line and confidence interval

    Parameters
    ----------
        x : array like, optional
            the 'x' value.
        y : array like, optional
        ci : optional
            confidence interval. Set to None if not required.
        show : bool, optional
            whether to show the plot or not
        annotation_key : str, optional
            The name of the value to annotate with.
        annotation_val : float, int, optional
            The value to annotate with.
        marker_size : int, optional
        line_color : optional
        marker_color: optional
        fill_color : optional
            only relevent if ci is not None.
        figsize : tuple, optional
        title : str, optional
            name to be used for title
        xlabel : str, optional
        ylabel : str, optional
        ax : plt.Axes, optional
            matplotlib axes to draw plot on. If not given, current avaialable
            will be used.

    Returns
    --------
    matplotlib.pyplot.Axes
        the matplotlib Axes on which regression plot is drawn.

    Examples
    --------
        >>> import numpy as np
        >>> from easy_mpl import regplot
        >>> x_, y_ = np.random.random(100), np.random.random(100)
        >>> regplot(x_, y_)

    Note
    ----
        If nans are present in x or y, they will be removed.

    """
    x = to_1d_array(x)
    y = to_1d_array(y)

    # remvoing nans based upon nans in x
    x_nan_idx = np.isnan(x)
    if x_nan_idx.sum() > 0:
        x = x[~x_nan_idx]
        y = y[~x_nan_idx]

    # remvoing nans based upon nans in y
    y_nan_idx = np.isnan(y)
    if y_nan_idx.sum() > 0:
        x = x[~y_nan_idx]
        y = y[~y_nan_idx]

    assert len(x) > 1, f"""
    length of x is smaller than 1 {x} 
    {len(y_nan_idx)} nans found in y and   {len(x_nan_idx)} nans found in x"""
    assert len(x) == len(y), f"x and y must be same length. Got {len(x)} and {len(y)}"

    mc, lc, fc = random.choice(regplot_combs)
    _metric_names = {'r2': '$R^2$'}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 5))

    ax.scatter(x, y, c=marker_color or mc,
               s=marker_size)  # set style options

    if annotation_key is not None:
        assert annotation_val is not None

        plt.annotate(f'{annotation_key}: {round(annotation_val, 3)}',
                     xy=(0.3, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top',
                     fontsize=16)
    _regplot(x,
             y,
             ax=ax,
             ci=ci,
             line_color=line_color or lc,
             fill_color=fill_color or fc)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=26)

    if show:
        plt.show()

    return ax
