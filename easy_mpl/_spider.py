
__all__ = ["spider_plot"]

import math
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import register_projections


def spider_plot(
        values:Union[pd.DataFrame, np.ndarray, list],
        tick_labels:list = None,
        highlight:Union[int, float]=None,
        plot_kws: Optional[Union[dict, List[dict]]] = None,
        xtick_kws: Optional[dict] = None,
        fill_kws : Optional[dict] = None,
        frame:str = "circle",
        color : Union[List[str], str] = None,
        fill_color: Union[List[str], str] = None,
        leg_kws : dict = None,
        labels: list = None,
        show: Optional[bool] = True,
        figsize=None,
)->plt.Axes:
    """
    Draws spider plot on an axes

    Parameters
    ----------
        values :
            values to display.
        tick_labels : list, optional
            tick labels. It's length should be equal to length of values
        plot_kws : dict, optional
            These can include, ``color``, ``linewidth``, ``linestyle`` etc
        xtick_kws : dict, optional
            These can include ``color``, ``size`` etc.
        fill_kws : dict, optional
            These can include ``color``, ``alpha`` etc.
        highlight : int/float optional (default=None)
            whether to highlight a certain circular line or not
        color : str
            colormap to use
        frame: str, optional (default="circle")
            whether the outer frame and grids should be polygon or circle
        figsize : tuple, optional (default=None)
            figure size
        fill_color :
            color to use for filling
        leg_kws : dict
            keyword arguments that will go to ax.legend()
        labels: list, optional (default=None)
            the labels for values
        show : bool, optional (default=True)
            whether to show the plot or not

    Returns
    -------
    plt.Axes
        matplotlib axes on which plot is drawn

    Examples
    --------
    >>> from easy_mpl import spider_plot
    >>> vals = [-0.2, 0.1, 0.0, 0.1, 0.2, 0.3]
    >>> spider_plot(values=vals)
    ... # specifying labels
    >>> labels = ['a', 'b','c', 'd', 'e', 'f']
    >>> spider_plot(values=vals, labels=labels)
    ... # specifying tick size
    >>> spider_plot(vals, labels, xtick_kws={'size': 13})
    ...
    >>> df = pd.DataFrame.from_dict(
    ... {'summer': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
    ... 'winter': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15,'f': 0.25}})
    >>> spider_plot(df, xtick_kws={'size': 13})
    ... # use polygon frame
    >>> spider_plot(values=vals, frame="polygon")

    """

    # todo, allow grids and frame to have different types

    if isinstance(values, (list, tuple)):
        values = np.array(values).reshape(-1, 1)

    if tick_labels is None:
        if isinstance(values, (pd.DataFrame, pd.Series)):
            tick_labels = values.index
        else:
            tick_labels = [f"F{i}" for i in range(len(values))]

    if labels is None:
        if isinstance(values, pd.Series):
            labels = [values.name]
        elif isinstance(values, pd.DataFrame):
            labels = values.columns.tolist()
        else:
            labels = [f'Value_{i}' for i in range(values.shape[1])]

    if isinstance(values, (pd.DataFrame, pd.Series)):
        values = values.values

    N = len(tick_labels)
    assert N == len(values)

    if not isinstance(plot_kws, list):
        plot_kws = [plot_kws for _ in range(N)]

    def_color = plt.cm.get_cmap('Set2', values.shape[1])

    if color is None:
        color = [def_color(i) for i in range(values.shape[1])]
    if fill_color is None:
        fill_color = [def_color(i) for i in range(values.shape[1])]

    if not isinstance(color, list):
        color = [color for _ in range(N)]

    if not isinstance(fill_color, list):
        fill_color = [fill_color for _ in range(N)]

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    if frame == "polygon":
        register_projections(N, frame)
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1,
                                subplot_kw= dict(projection='radar'))
    else:
        ax = plt.subplot(polar=True)

    _xtick_kws = {'color': 'grey', 'size': 14}
    xtick_kws = xtick_kws or {}
    _xtick_kws.update(xtick_kws)

    plt.xticks(angles[:-1], tick_labels, **_xtick_kws)

    for idx in range(values.shape[1]):

        plot_kws_ = plot_kws[idx]

        _plot_kws = {"color": color[idx], "linewidth": 2, "linestyle": 'solid'}
        plot_kws_ = plot_kws_ or {}
        _plot_kws.update(plot_kws_)

        val = values[:, idx].tolist()
        val.append(val[0])
        ax.plot(angles, val, **_plot_kws)

        _fill_kws = {"color":fill_color[idx], "alpha":.4, 'label': '_nolegend_'}
        fill_kws = fill_kws or {}
        _fill_kws.update(fill_kws)
        ax.fill(angles, val, **_fill_kws)

    if highlight:
        val_minus = highlight - 0.005
        val_plus = highlight + 0.005
        ax.fill_between(np.linspace(0, 2 * np.pi, 100),
                        val_minus,
                        val_plus,
                        color='red',
                        zorder=10)

    plt.gca().set_rmax(np.max(values) + np.max(values)*0.3)

    _leg_kws = {'labelspacing': 0.1, 'fontsize': 12, 'bbox_to_anchor': (1.3, 1.1)}
    leg_kws = leg_kws or {}
    _leg_kws.update(leg_kws)
    legend = ax.legend(labels, **_leg_kws)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
