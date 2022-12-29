
__all__ = ["bar_chart"]

import random
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import KeysView, ValuesView

from .utils import BAR_CMAPS
from .utils import process_axes
from .utils import create_subplots
from .utils import to_1d_array, make_cols_from_cmap


def bar_chart(
        values,
        labels=None,
        orient:str = 'h',
        sort:bool = False,
        max_bars:int = None,
        errors = None,
        color=None,
        cmap: Union[str, List[str]] = None,
        rotation:int = 0,
        bar_labels: Union[list, np.ndarray] = None,
        bar_label_kws=None,
        share_axes: bool = True,
        width = None,
        ax:plt.Axes = None,
        ax_kws: dict = None,
        show:bool = True,
        **kwargs
) -> Union[plt.Axes, List[plt.Axes]]:
    """
    plots bar chart

    Parameters
    -----------
        values : array like
            array like e.g. list/numpy array/ pandas series/ pandas dataframe / tuple
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
        bar_labels : list
            labels of the bars
        bar_label_kws : dict
            keyword arguments for :obj:`matplotlib.axes.Axes.bar_label`
        share_axes : bool (default=True)
            only relevant if given array is 2 dimentional. In such a case,
            this refers to whether to draw two bar charts on same axes or on two
            separate axes
        width : float
            width of the bars
        ax : :obj:`matplotlib.axes`, optional
            If not given, current available axes will be used
        ax_kws : dict, optional
            any keyword arguments for processing of axes that will go to
            :py:func:`easy_mpl.utils.process_axes`
        show : bool, optional
            whether to show the plot or not
        **kwargs :
            any additional keyword arguments for :obj:`matplotlib.axes.Axes.bar`
            or :obj:`matplotlib.axes.Axes.barh`

    Returns
    --------
    :obj:`matplotlib.axes`
        matplotlib Axes  or list of matplotlib Axes on which the bar_chart is drawn.
        If ``show`` is False, this axes can be used for further processing

    Examples
    --------
        >>> from easy_mpl import bar_chart
        >>> bar_chart([1,2,3,4,4,5,3,2,5])
        specifying labels
        >>> bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])
        sorting the data
        >>> bar_chart([1,2,3,4,4,5,3,2,5], sort=True)
        multiple bar charts
        >>> bar_chart(np.random.randint(0, 10, (5, 2)), color=['salmon', 'cadetblue'])

    See :ref:`sphx_glr_auto_examples_bar_chart.py` for more examples

    .. _axes.bar:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html

    .. _axes.barh:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html
    """

    if labels is None:
        if hasattr(values, "index") and hasattr("values", "name"):
            labels = values.index

    naxes = 1
    ncharts = 1
    if is_1d(values):
        values = to_1d_array(values)
    else:
        values = np.array(values)
        ncharts = values.shape[1]
        if share_axes:
            kwargs['edgecolor'] = kwargs.get('edgecolor', 'k')
        else:
            naxes = values.shape[1]

    colors = get_color(cmap, color, ncharts, len(values))

    figsize = None
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')

    ax = maybe_create_axes(ax, naxes, figsize=figsize)

    if ncharts == 1:
        values, labels, bar_labels = preprocess(values, labels,
                                                bar_labels, sort, max_bars)
        ind = np.arange(len(values))
        bar_on_axes(ax[0], orient=orient, ax_kws=ax_kws, ind=ind,
                    values=values,
                    width=width, ticks=ind, labels=labels, color=colors[0],
                    bar_labels=bar_labels,
                   rotation=rotation, errors=errors,
                    bar_label_kws=bar_label_kws, kwargs=kwargs)

    elif share_axes:
        ind = np.arange(len(values))  # the label locations
        width = width or 1/ncharts * 0.9  # the width of the bars

        inds = []
        for idx in range(ncharts):
            if idx>0:
                ind = ind + width
            inds.append(ind)
        inds = np.column_stack(inds)
        ticks = np.mean(inds, axis=1)

        for idx in range(ncharts):

            _kwargs =kwargs.copy()
            _kwargs['label'] = _kwargs.get('label', idx)

            vals, labels, bar_labels = preprocess(values[:, idx], labels,
                                                  bar_labels, sort, max_bars)
            bar_on_axes(ax[0], orient, ax_kws,
                        inds[:, idx], vals, width, ticks, labels,
                        colors[idx], bar_labels,
                       rotation, errors, bar_label_kws, _kwargs)

    else:
        for idx in range(naxes):
            axes = ax[idx]
            data = values[:, idx]
            data, labels, bar_labels = preprocess(data, labels, bar_labels,
                                                  sort, max_bars)

            _kwargs = kwargs.copy()
            _kwargs['label'] = _kwargs.get('label', idx)

            ind = np.arange(len(data))
            bar_on_axes(axes, orient, ax_kws,
                        ind, data, width, ind, labels,
                        colors[idx], bar_labels,
                        rotation, errors,
                        bar_label_kws=bar_label_kws, kwargs=_kwargs)

    if show:
        plt.show()

    if len(ax) == 1:
        ax = ax[0]

    return ax


def maybe_create_axes(ax, naxes:int, figsize=None)->List[plt.Axes]:
    if ax is None:
        ax = plt.gca()
        if naxes>1:
            f, ax = create_subplots(ax=ax, naxes=naxes, figsize=figsize)
            ax = ax.flatten()
        else:
            if figsize:
                ax.figure.set_size_inches(figsize)
            ax = [ax]
    elif naxes>1:
        f, ax = create_subplots(ax=ax, naxes=naxes, figsize=figsize)
        ax = ax.flatten()
    else:
        if figsize:
            ax.figure.set_size_inches(figsize)
        ax = [ax]

    return ax


def handle_sort(sort, values, labels, bar_labels):
    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]
        if bar_labels is not None:
            bar_labels = np.array(bar_labels)
            bar_labels = bar_labels[sort_idx]
            if 'float' in bar_labels.dtype.name:
                bar_labels = np.round(bar_labels, decimals=2)

    return values, labels, bar_labels


def handle_maxbars(max_bars, values, labels):
    if max_bars:
        n = len(values) - max_bars
        last_val = sum(values[0:-max_bars])
        values = values[-max_bars:]
        labels = labels[-max_bars:]
        values = np.append(last_val, values)
        labels = np.append(f"Rest of {n}", labels)
    return values, labels


def preprocess(values, labels, bar_labels, sort, max_bars):
    if labels is None:
        labels = [f"F{i}" for i in range(len(values))]

    values, labels, bar_labels = handle_sort(sort, values, labels, bar_labels)

    values, labels = handle_maxbars(max_bars, values, labels)

    return values, labels, bar_labels


def bar_on_axes(ax, orient, ax_kws, *args, **kwargs):
    if orient in ['h', 'horizontal']:
        horizontal_bar(ax, *args, **kwargs)
    else:
        vertical_bar(ax, *args, **kwargs)

    if ax_kws:
        process_axes(ax, **ax_kws)

    return


def horizontal_bar(ax, ind, values, width, ticks, labels, color, bar_labels,
                   rotation, errors, bar_label_kws, kwargs):

    if width:
        bar = ax.barh(ind, values, width, color=color, **kwargs)
    else:
        bar = ax.barh(ind, values, color=color, **kwargs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=rotation)

    set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   values, ind)

    if 'label' in kwargs:
        ax.legend()
    return


def vertical_bar(ax, ind, values, width, ticks, labels, color, bar_labels,
                 rotation, errors, bar_label_kws, kwargs):

    bar = ax.bar(ind, values, width=width or 0.8, color=color, **kwargs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation)

    set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   ind, values)
    return


def set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   values, ind):
    if bar_labels is not None:
        bar_label_kws = bar_label_kws or {'label_type': 'center'}
        if hasattr(ax, 'bar_label'):
            ax.bar_label(bar, labels=bar_labels, **bar_label_kws)
        else:
            bar.set_label(bar_labels)

    if errors is not None:
        ax.errorbar(values, ind, xerr=errors, fmt=".",
                    color="black")
    return


def is_1d(array):
    if isinstance(array, (KeysView, ValuesView)):
        array = np.array(list(array))
    else:
        array = np.array(array)
    if len(array)==array.size:
        return True
    return False


def get_color(cmap, color, ncharts, n_bars):
    if not isinstance(cmap, list):
        cmap = [cmap for _ in range(ncharts)]

    if not isinstance(color, list):
        color = [color for _ in range(ncharts)]

    colors = []
    for idx in range(ncharts):

        cm = make_cols_from_cmap(cmap[idx] or random.choice(BAR_CMAPS), n_bars, 0.2)

        clr = color[idx] if color[idx] is not None else cm

        colors.append(clr)

    return colors
