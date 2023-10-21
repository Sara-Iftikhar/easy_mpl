
__all__ = ["regplot"]

import random
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
from easy_mpl.utils import process_axes
from easy_mpl.utils import AddMarginalPlots


REGC_COMBS = [
    ['cadetblue', 'slateblue', 'darkslateblue'],
    ['cadetblue', 'mediumblue', 'mediumblue'],
    ['cornflowerblue', 'dodgerblue', 'darkblue'],
    ['cornflowerblue', 'dodgerblue', 'steelblue'],
    ['cornflowerblue', 'mediumblue', 'dodgerblue'],
    ['cornflowerblue', 'steelblue', 'mediumblue'],
    ['darkslateblue', 'aliceblue', 'mediumblue'],
    ['darkslateblue', 'blue', 'royalblue'],
    ['darkslateblue', 'blueviolet', 'royalblue'],
    ['darkslateblue', 'darkblue', 'midnightblue'],
    ['darkslateblue', 'mediumblue', 'darkslateblue'],
    ['darkslateblue', 'midnightblue', 'mediumblue'],
    ['seagreen', 'darkslateblue', 'cadetblue'],
    ['cadetblue', 'darkblue', 'midnightblue'],
    ['cadetblue', 'deepskyblue', 'cadetblue']
]


def regplot(
        x: Union[np.ndarray,  list],
        y: Union[np.ndarray, list],
        label:str = None,
        marker_size: Union[int, float] = 20,
        marker_color=None,
        scatter_kws: dict = None,
        line_style='-',
        line_color=None,
        line_kws: dict = None,
        ci: Union[int, None] = 95,
        fill_color=None,
        figsize: tuple = None,
        marginals: bool = False,
        marginal_ax_size: Union[float, List[float]] = 0.7,
        marginal_ax_pad: Union[float, List[float]] = 0.25,
        ridge_line_kws: Union[dict, List[dict]] = None,
        fill_kws: Union[dict, List[dict]] = None,
        hist: bool = True,
        hist_kws: Union[dict, List[dict]] = None,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        show: bool = True,
) -> plt.Axes:
    """
    Regpression plot with regression line and confidence interval

    Parameters
    ----------
        x : array like, optional
            the 'x' value. It can be numpy array, pandas DataFram/Series or a list
        y : array like, optional
             It can be numpy array, pandas DataFram/Series or a list
        ci : int, optional
            confidence interval. Set to None if not required.
        label : str, optional
            Label to use as legend.
            The value to annotate with.
        marker_size : int, optional
            size of marker
        marker_color: optional
            color of marker
        scatter_kws : dict
            keyword arguments for :obj:`matplotlib.axes.scatter`
        line_style : str (default='-')
            line style will be used as ax.plot(x,y,line_style)
            Set this to None if you don't want to plot line
        line_color : optional
            color of line
        line_kws : dict
            keyword arguments for axes.plot for line plot
        fill_color : optional
            only relevent if ci is not None.
        figsize : tuple, optional
            figure size (width, height)
        marginals : bool (default=False)
            whether to draw the marginal plots or not. If yes, marginal plots
            will
            be drawn using :py:class:`easy_mpl.utils.AddMarginalPlots`
        marginal_ax_size :
            size of marginal axes. If given as list, the first argument is taken
            for
            horizontal marginal axes and second argument for vertical merginal
            axees.
            It is only valid if ``marginals`` is set to True.
        marginal_ax_pad :
            pad value for marginal axes. If given as list, the first argument
            is taken for
            horizontal marginal axes and second argument for vertical merginal
            axees.
            It is only valid if ``marginals`` is set to True.
        ridge_line_kws :
            keyword arguments for a:obj:`matplotlib.axes.Axes.plot to draw the
             kde.
            It can be dictionary
            or list of two dictionaries, where first dictionary is for hoziontal
            marginal axes and second dictionary is for vertical marginal axes
        fill_kws : dict/List[dict] (default=None)
            keyword arguments for :obj:`matplotlib.axes.Axes.fill_between`. It
            can be dictionary or list of
            two dictionaries, where first dictionary is for hoziontal
            marginal axes and second dictionary is for vertical marginal axes
        hist : bool
            whether to draw the histogram or not on marginal axes
        hist_kws :
            keyword arguments for :obj:`matplotlib.axes.Axes.hist`. It can be
            dictionary or list of
            two dictionaries, where first dictionary is for hoziontal
            marginal axes and second dictionary is for vertical marginal axes
        ax : plt.Axes, optional
            matplotlib axes :obj:`matplotlib.axes` to draw plot on. If not given,
            current avaialable will be used.
        ax_kws : dict (default=None)
            keyword arguments for :py:func:`easy_mpl.utils.process_axes`
        show : bool, optional
            whether to show the plot or not

    Returns
    --------
    matplotlib.pyplot.Axes
        the matplotlib Axes :obj:`matplotlib.axes` on which regression plot is
        drawn.

    Examples
    --------
        >>> import numpy as np
        >>> from easy_mpl import regplot
        >>> x_, y_ = np.random.random(100), np.random.random(100)
        >>> regplot(x_, y_)

    See :ref:`sphx_glr_auto_examples_reg_plot.py` for more examples

    Note
    ----
        If nans are present in x or y, they will be removed.

    """
    x = np.array(x).reshape(-1,)
    y = np.array(y).reshape(-1,)

    # removing nans based upon nans in x
    x_nan_idx = np.isnan(x)
    if x_nan_idx.sum() > 0:
        x = x[~x_nan_idx]
        y = y[~x_nan_idx]

    # removing nans based upon nans in y
    y_nan_idx = np.isnan(y)
    if y_nan_idx.sum() > 0:
        x = x[~y_nan_idx]
        y = y[~y_nan_idx]

    assert len(x) > 1, f"""
    length of x is smaller than 1 {x} 
    {len(y_nan_idx)} nans found in y and   {len(x_nan_idx)} nans found in x"""
    assert len(x) == len(y), f"""
    x and y must be same length. Got {len(x)} and {len(y)}"""

    mc, lc, fc = random.choice(REGC_COMBS)
    _metric_names = {'r2': '$R^2$'}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 5))

    _scatter_kws = dict()
    if label:
        _scatter_kws['label'] = label

    if scatter_kws is not None:
        _scatter_kws.update(scatter_kws)

    if marker_color is None:
        marker_color = mc

    ax.scatter(x, y, c=marker_color,
               s=marker_size, **_scatter_kws)

    if label:
        ax.legend()

    if line_kws is None:
        line_kws = dict()

    if line_color is None:
        line_color = lc

    if fill_color is None:
        fill_color = fc

    _regplot(x,
             y,
             ax=ax,
             ci=ci,
             line_style=line_style,
             line_color=line_color,
             fill_color=fill_color,
             **line_kws)

    if marginals:
        AddMarginalPlots(ax,
                         pad=marginal_ax_pad,
                         size=marginal_ax_size,
                         hist=hist,
                         hist_kws=hist_kws,
                         ridge_line_kws=ridge_line_kws,
                         fill_kws=fill_kws)(x, y)

    _ax_kws = {'xlabel': 'Observed',
               'xlabel_kws': {'fontsize': 14},
               'ylabel': 'Prediction',
               'ylabel_kws': {'fontsize': 14},
               }

    if ax_kws is not None:
        _ax_kws.update(ax_kws)

    process_axes(ax, **_ax_kws)

    if show:
        plt.show()

    return ax


def bootdist(f, args, n_boot=1000, **func_kwargs):

    n = len(args[0])
    integers = np.random.randint
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(f(*sample, **func_kwargs))

    return np.array(boot_dist)


def _regplot_paras(x, y, ci: int = None):
    """prepares parameters for regplot"""
    grid = np.linspace(np.min(x), np.max(x), 100)
    x = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(x, y))

    err_bands = None
    if ci:
        boots = bootdist(reg_func, args=[x, y], n_boot=1000).T

        yhat_boots = grid.dot(boots).T
        err_bands = _ci(yhat_boots, ci, axis=0)

    return grid, yhat, err_bands


def _ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def reg_func(_x, _y):
    return np.linalg.pinv(_x).dot(_y)


def _regplot(x, y, ax, ci=None,
             line_style="-",
             line_color=None, fill_color=None, **kwargs):

    grid, yhat, err_bands = _regplot_paras(x, y, ci)

    if line_style is not None:
        ax.plot(grid[:, 1], yhat, line_style, color=line_color, **kwargs)

        if ci:
            ax.fill_between(grid[:, 1], *err_bands,
                            facecolor=fill_color,
                            alpha=.15,
                            where=np.array([True for _ in range(len(grid))])
                        )
    return ax
