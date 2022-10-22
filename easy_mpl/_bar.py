
__all__ = ["bar_chart"]

import random
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import BAR_CMAPS
from .utils import process_axis
from .utils import to_1d_array, make_cols_from_cmap


def bar_chart(
        values,
        labels=None,
        orient='h',
        sort=False,
        max_bars:int = None,
        errors: Union = None,
        color=None,
        cmap: str = None,
        rotation=0,
        show=True,
        ax=None,
        bar_labels: Union[list, np.ndarray] = None,
        bar_label_kws=None,
        ax_kws: dict = None,
        **kwargs
) -> plt.Axes:
    """
    plots bar chart

    Parameters
    -----------
        values : array like
            1 dimensional array or list
        labels : list, optional
            used for labeling each bar
        orient : `str`, optional
            orientation of bars. either 'h' or 'v'
        sort : bool, optional (default=None)
            whether to sort the bars based upon their values or not
        max_bars: int, optional (default=None)
            maximum number of bars to show
        errors : list, optional
            for error bars
        color : bool, optional (default=None)
            color for bars. It can any color value valid for matplotlib.
        cmap : str, optional (default=None)
            matplotlib colormap
        rotation : int, optional
            rotation angle of ticklabels
        ax : plt.Axes, optional
            If not given, current available axes will be used
        show : bool, optional,
        bar_labels : list
            labels of the bars
        bar_label_kws : dict
        ax_kws : dict, optional
            any keyword arguments for processing of axes that will go to
            :py:func:`easy_mpl.utils.process_axes`
        **kwargs :
            any additional keyword arguments for `axes.bar`_ or `axes.barh`_

    Returns
    --------
    matplotlib.pyplot.Axes
        matplotlib Axes on which the bar_chart is drawn. If ``show`` is False, this axes
        can be used for further processing

    Example
    --------
        >>> from easy_mpl import bar_chart
        >>> bar_chart([1,2,3,4,4,5,3,2,5])
        specifying labels
        >>> bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])
        sorting the data
        >>> bar_chart([1,2,3,4,4,5,3,2,5], sort=True)


    .. _axes.bar:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html

    .. _axes.barh:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html
    """

    if labels is None:
        if hasattr(values, "index"):
            labels = values.index

    values = to_1d_array(values)

    cmap = make_cols_from_cmap(cmap or random.choice(BAR_CMAPS), len(values), 0.2)
    color = color if color is not None else cmap

    if not ax:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    if labels is None:
        labels = [f"F{i}" for i in range(len(values))]

    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]

    if max_bars:
        n = len(values) - max_bars
        last_val = sum(values[0:-max_bars])
        values = values[-max_bars:]
        labels = labels[-max_bars:]
        values = np.append(last_val, values)
        labels = np.append(f"Rest of {n}", labels)

    if orient in ['h', 'horizontal']:
        bar = ax.barh(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, rotation=rotation)

        if bar_labels is not None:
            bar_label_kws = bar_label_kws or {'label_type': 'center'}
            if hasattr(ax, 'bar_label'):
                ax.bar_label(bar, labels=bar_labels, **bar_label_kws)
            else:
                bar.set_label(bar_labels)

        if errors is not None:
            ax.errorbar(values, np.arange(len(values)), xerr=errors, fmt=".",
                        color="black")
    else:
        bar = ax.bar(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=rotation)

        if bar_labels is not None:
            bar_label_kws = bar_label_kws or {'label_type': 'center'}
            if hasattr(ax, 'bar_label'):
                ax.bar_label(bar, labels=bar_labels, **bar_label_kws)
            else:
                bar.set_label(bar_labels)

        if errors is not None:
            ax.errorbar(np.arange(len(values)), values, yerr=errors, fmt=".",
                        color="black")

    if ax_kws:
        ax = process_axis(ax, **ax_kws)

    if 'label' in kwargs:
        ax.legend()

    if show:
        plt.show()

    return ax
