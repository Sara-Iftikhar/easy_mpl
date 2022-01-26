
__all__ = [
    "plot",
    "bar_chart",
    "regplot",
    "imshow",
    "hist",
    "pie",
    "scatter",
    "contour",
    "dumbbell_plot"
]

import random
from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .utils import to_1d_array, get_cmap, _regplot, process_axis
from .utils import BAR_CMAPS, regplot_combs


def bar_chart(
        values,
        labels=None,
        orient='h',
        sort=False,
        color=None,
        xlabel=None,
        xlabel_fs=None,
        title=None,
        title_fs=None,
        show_yaxis=True,
        rotation=0,
        show=True,
        ax=None,
        **kwargs
):
    """
    plots bar chart

    Parameters
    -----------
        values : array like
        labels : list, optional
        orient : `str`, optional
            orientation of bars. either 'h' or 'v'
        sort : bool, optional
            whether to sort the bars based upon their values or not
        color : bool, optional
        xlabel : str, optional
        xlabel_fs : int, optional
            xlabel font size
        title : str, optional
        title_fs : int, optional
            title font size
        show_yaxis : bool, optional
        rotation : int, optional
        ax : plt.Axes, optional
            If not given, current available axes will be used
        show : bool, optional
        **kwargs :
            any additional keyword arguments for `axes.bar`_ or `axes.barh`_

    Example
    --------
        >>> from easy_mpl import bar_chart
        >>> bar_chart([1,2,3,4,4,5,3,2,5])
        specifying labels
        >>> bar_chart([3,4,2,5,10], ['a', 'b', 'c', 'd', 'e'])
        sorting the data
        >>> bar_chart([1,2,3,4,4,5,3,2,5], sort=True)

    # todo, add labeling of bars as optional parameter

    .. _axes.bar:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html

    .. _axes.barh:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html
    """

    values = np.array(values)

    cm = get_cmap(random.choice(BAR_CMAPS), len(values), 0.2)
    color = color if color is not None else cm

    if not ax:
        _, ax = plt.subplots()

    if labels is None:
        labels = [f"F{i}" for i in range(len(values))]

    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]

    if orient == 'h':
        ax.barh(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, rotation=rotation)

    else:
        ax.bar(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=rotation)

    if not show_yaxis:
        ax.get_yaxis().set_visible(False)

    if xlabel:
        ax.set_xlabel(xlabel, fontdict={'fontsize': xlabel_fs})

    if title:
        ax.set_title(title, fontdict={'fontsize': title_fs})

    if show:
        plt.show()

    return ax


def plot(
        *args,
        show: bool = True,
        **kwargs
)->plt.Axes:
    """
    One liner plot function. It's use is not more complex than `axes.plot()`_ or
    `plt.plot()`_ . However it accomplishes all in one line what requires multiple
    lines in matplotlib. args and kwargs can be anything which goes into `plt.plot()`_
    or `axes.plot()`_.

    Parameters
    ----------
        *args :
            either a single array or x and y arrays or anything which can go to
            `axes.plot()`_ or anything which can got to `plt.plot()`_ .
        show : bool, optional
        **kwargs : optional
            Anything which goes into `easy_mpl.utils.process_axis`.

    Returns
    -------
        matplotlib.pyplot.Axes

    Example
    --------
        >>> from easy_mpl import plot
        >>> import numpy as np
        >>> plot(np.random.random(100))
        use x and y
        >>> plot(np.arange(100), np.random.random(100))
        use x and y
        >>> plot(np.arange(100), np.random.random(100), '.')
        string after arrays represent marker style
        >>> plot(np.random.random(100), '.')
        use cutom marker
        >>> plot(np.random.random(100), '--*')
        using label keyword
        >>> plot(np.random.random(100), '--*', label='label')
        log transform y-axis
        >>> plot(np.random.random(100), '--*', logy=True, label='label')

    .. _axes.plot():
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

    .. _plt.plot():
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    """

    plot_kwargs = ('linewidth', 'linestyle', 'marker', 'fillstyle', 'ms', 'color',
                   'drawstyle', 'y_data', 'url', 'mfc', 'mec', 'snap'
                   'lw', 'ls', 'ds'
                   )
    _plot_kwargs = {}
    for arg in plot_kwargs:
        if arg in kwargs:
            _plot_kwargs[arg] = kwargs.pop(arg)

    plot_args = []

    marker = None
    if len(args)==1:
        data, = args
        data = [data]
    elif len(args) == 2 and not isinstance(args[1], str):
        data = args
    elif len(args) == 2 and isinstance(args[1], str):
        data, marker = args[0], args[1]
        data = [data]
    elif len(args)==3:
        *data, marker = args
        if isinstance(marker, np.ndarray):
            data.append(marker)
            marker = None
    else:
        data = args

    if marker:
        plot_args.append(marker)
        assert 'marker' not in _plot_kwargs  # we have alreay got marker

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()

    # if log_nz:
    #     data = deepcopy(data)
    #     _data = data.values
    #     d_nz_idx = np.where(_data > 0.0)
    #     data_nz = _data[d_nz_idx]
    #     d_nz_log = np.log(data_nz)
    #     _data[d_nz_idx] = d_nz_log
    #     _data = np.where(_data < 0.0, 0.0, _data)
    #     data = pd.Series(_data, index=data.index)
    s = data[0]
    if isinstance(s, pd.Series):
        kwargs['min_xticks'] = kwargs.get('min_xticks', 3)
        kwargs['max_xticks'] = kwargs.get('max_xticks', 5)
        kwargs['xlabel'] = kwargs.get('xlabel', s.index.name)
        kwargs['ylabel'] = kwargs.get('ylabel', s.name)
    elif isinstance(s, pd.DataFrame):
        kwargs['min_xticks'] = kwargs.get('min_xticks', 3)
        kwargs['max_xticks'] = kwargs.get('max_xticks', 5)
        if  s.shape[1] == 1:
            kwargs['xlabel'] = kwargs.get('xlabel', s.index.name)
            kwargs['ylabel'] = kwargs.get('ylabel', s.columns.tolist()[0])
        else:
            kwargs['xlabel'] = kwargs.get('xlabel', s.index.name)
            kwargs['label'] = kwargs.get('label', s.columns.tolist())
            for col in s.columns:
                _plot_kwargs['label'] = col
                data[0] = s[col]
                ax.plot(*data, *plot_args, **_plot_kwargs)
            return _process_axis(ax, show, kwargs)

    _plot_kwargs['label'] = kwargs.get('label', None)

    ax.plot(*data, *plot_args, **_plot_kwargs)
    return _process_axis(ax, show, kwargs)


def _process_axis(ax, show, kwargs):
    if kwargs:
        ax = process_axis(ax=ax, **kwargs)
    if kwargs.get('save', False):
        plt.savefig(f"{kwargs.get('name', 'fig.png')}")
    if show:
        plt.show()
    return ax


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
)->plt.Axes:
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

    Examples
    --------
        >>> import numpy as np
        >>> from easy_mpl import regplot
        >>> x_, y_ = np.random.random(100), np.random.random(100)
        >>> regplot(x_, y_)

    """
    x = to_1d_array(x)
    y = to_1d_array(y)

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


def imshow(
        values,
        xlabel=None,
        title=None,
        ylabel=None,
        yticklabels=None,
        xticklabels=None,
        show=True,
        annotate=False,
        annotate_kws=None,
        colorbar:bool=False,
        colorbar_orientation:str = 'vertical',
        ax=None,
        **kwargs
)->tuple:
    """
    One stop shop for matplotlib's imshow function

    Parameters
    ----------
        values: 2d array
        xlabel:  str, optional
        ylabel : str, optional
        title : str, optional
        show : bool, optional
            whether to show the plot or not
        annotate : bool, optional
        annotate_kws : dict, optional
        colorbar : bool, optional
            whether to draw colorbar or not
        colorbar_orientation : str, optional
            either "vertical" or "horizontal"
        xticklabels : optional
        yticklabels :  optional
        ax : plt.Axes, optional
            if not given, current available axes will be used
        **kwargs : optional
            any further keyword arguments for `axes.imshow`_

    Returns
    -------
        a tuple whose first vlaue is matplotlib axes and second argument is AxesImage

    Examples
    --------
        >>> import numpy as np
        >>> from easy_mpl import imshow
        >>> x = np.random.random((10, 5))
        >>> imshow(x, annotate=True)
        # show colorbar
        >>> imshow(x, colorbar=True)

    .. _axes.imshow:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
    """

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(values, **kwargs)

    if annotate:
        annotate_kws = annotate_kws or {"color": "w", "ha":"center", "va":"center"}
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                _ = ax.text(j, i, round(values[i, j], 2),
                            **annotate_kws)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)

    process_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    if colorbar:
        fig: plt.Figure = plt.gcf()
        fig.colorbar(im, orientation=colorbar_orientation, pad=0.2)

    if show:
        plt.show()

    return ax, im


def hist(
        x:Union[list, np.ndarray, pd.Series, pd.DataFrame],
        hist_kws: dict = None,
        grid:bool = True,
        ax: plt.Axes = None,
        show: bool = True,
        **kwargs
)->plt.Axes:
    """
    one stop shop for histogram

    Parameters
    -----------
        x : list, array, optional
            array like, must not be greader than 1d
        grid : bool, optional
            whether to show the grid or not
        show : bool, optional
            whether to show the plot or not
        ax : plt.Axes, optional
            axes on which to draw the plot
        hist_kws : dict, optional
            any keyword arguments for `axes.hist`_
        **kwargs : optional
            any keyword arguments for axes manipulation such as title, xlable, ylable etc

    Returns
    --------
        matplotlib Axes

    Example
    --------
        >>> from easy_mpl import hist
        >>> hist(np.random.random((10, 1)))

    .. _axes.hist:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    """

    if not ax:
        ax = plt.gca()
    hist_kws = hist_kws or {}
    n, bins, patches = ax.hist(x, **hist_kws)

    process_axis(ax, grid=grid, **kwargs)

    if show:
        plt.show()

    return ax


def pie(
        vals: Union[list, np.ndarray, pd.Series] = None,
        fractions: Union[list, np.ndarray, pd.Series] = None,
        labels: list = None,
        ax: plt.Axes = None,
        title: str = None,
        name: str = None,
        save: bool = True,
        show: bool = True,
        **kwargs
)->plt.Axes:
    """
    pie chart

    Parameters
    ----------
        vals : array like,
            unique values and their counts will be inferred from this array.
        fractions : list, array, optional
            if given, vals must not be given
        labels : list, array, optional
            labels for unique values in vals, if given, must be equal to unique vals
            in vals. Otherwise "unique_value (counts)" will be used for labeling.
        ax : plt.Axes, optional
            the axes on which to draw, if not given current active axes will be used
        title: str, optional
            if given, will be used for title
        name: str, optional
        save: bool, optional
        show: bool, optional
        **kwargs: optional
            any keyword argument will go to `axes.pie`_

    Returns
    -------
        a matplotlib axes. This can be used for further processing by making show=False.

    Example
    -------
        >>> pie(np.random.randint(0, 3, 100))
        or by directly providing fractions
        >>> pie([0.2, 0.3, 0.1, 0.4])
        # to explode 0.3
        >>> explode = (0, 0.1, 0, 0, 0)
        >>> pie(fractions=[0.2, 0.3, 0.15, 0.25, 0.1], explode=explode)

    .. _axes.pie:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
    """
    #todo, add example for explode and partial pie chart
    if ax is None:
        ax = plt.gca()

    if fractions is None:
        fractions = pd.Series(vals).value_counts(normalize=True).values
        vals = pd.Series(vals).value_counts().to_dict()
        if labels is None:
            labels = [f"{value} ({count}) " for value, count in vals.items()]
    else:
        assert vals is None
        if labels is None:
            labels = [f"f{i}" for i in range(len(fractions))]

    if 'autopct' not in kwargs:
        kwargs['autopct'] = '%1.1f%%'

    ax.pie(fractions,
           labels=labels,
           **kwargs)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if title:
        plt.title(title, fontsize=20)

    if save:
        name = name or "pie.png"
        plt.savefig(name, dpi=300)

    if show:
        plt.show()

    return ax


def scatter(
        x,
        y,
        colorbar: bool = False,
        colorbar_orientation: str = "vertical",
        show: bool = True,
        ax: plt.Axes = None,
        **kwargs
)->Tuple[plt.Axes, mpl.collections.PathCollection]:
    """
    scatter plot between two arrays x and y

    Parameters
    ----------
    x : list, array
    y : list, array
    colorbar : bool, optional
    colorbar_orientation : str, optional
    show : bool, optional
        whether to show the plot or not
    ax : plt.Axes, optional
        if not given, current available axes will be used
    **kwargs : optional
        any additional keyword arguments for `axes.scatter`_

    Returns
    --------
    matplotlib Axes
    matplotlib.collections.PathCollection

    Examples
    --------
        >>> from easy_mpl import scatter
        >>> import numpy as np
        >>> x_ = np.random.random(100)
        >>> y_ = np.random.random(100)
        >>> scatter(x_, y_, show=False)
        show colorbar
        >>> scatter(x_, y_, colorbar=True, show=False)
        retrieve axes for further processing
        >>> axes, _ = scatter(x_, y_, show=False)
        >>> assert isinstance(axes, plt.Axes)

    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

    """
    if ax is None:
        ax = plt.gca()

    x = to_1d_array(x)
    y = to_1d_array(y)

    if colorbar and 'c' not in kwargs:
        kwargs['c'] = np.arange(len(x))

    sc = ax.scatter(x, y, **kwargs)

    if colorbar:
        fig: plt.Figure = plt.gcf()
        fig.colorbar(sc, orientation=colorbar_orientation, pad=0.1)

    if show:
        plt.show()

    return ax, sc


def contour(
        x,
        y,
        z,
        fill_between=False,
        show_points=False,
        colorbar=True,
        label_contours=False,
        contour_kws: dict = None,
        fill_between_kws: dict = None,
        show_points_kws: dict = None,
        label_contour_kws: dict = None,
        ax=None,
        show=True,
        **kwargs
):
    """A contour plot of irregularly spaced data coordinates.

    Parameters
    ---------
        x : array, list
            a 1d array defining grid positions along x-axis
        y : array, list
            a 1d array defining grid positions along y-axis
        z : array, list
            values on grids/points defined by x and y
        fill_between : bool, optional
            whether to fill the space between colors or not
        show_points : bool, optional
            whether to show the
        label_contours : bool, optional
            whether to label the contour lines or not
        colorbar : bool, optional
            whether to show the colorbar or not. Only valid if `fill_between` is
            True
        contour_kws : dict, optional
            any keword argument for `axes.tricontour`_
        fill_between_kws : dict, optional
            any keword argument for `axes.tricontourf`_
        show_points_kws : dict, optional
            any keword argument for `axes.plot`_
        label_contour_kws : dict, optional
            any keword argument for `axes.clabel`_
        ax : plt.Axes, optional
            matplotlib axes to work on. If not given, current active axes will
            be used.
        show : bool, optional
            whether to show the plot or not
        **kwargs : optional
            any keyword arguments for easy_mpl.utils.process_axis

    Returns
    -------
        a matplotliblib Axes

    Examples
    --------
        >>> from easy_mpl import contour
        >>> import numpy as np
        >>> _x = np.random.uniform(-2, 2, 200)
        >>> _y = np.random.uniform(-2, 2, 200)
        >>> _z = _x * np.exp(-_x**2 - _y**2)
        >>> contour(_x, _y, _z, fill_between=True, show_points=True)
        show contour labels
        >>> contour(_x, _y, _z, label_contours=True, show_points=True)

    Note
    ----
    The length of x and y should be same. The actual grid is created using axes.tricontour
    and axes.tricontourf functions.

    .. _axes.tricontour:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontour.html

    .. _axes.tricontourf:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tricontourf.html

    .. _axes.plot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

    .. _axes.clabel:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.clabel.html
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert len(x) == len(y) == len(z)

    if ax is None:
        ax = plt.gca()

    kws = contour_kws or {"levels": 14, "linewidth": 0.5, "colors": "k"}
    CS = ax.tricontour(x, y, z, **kws)

    if fill_between:
        kws = fill_between_kws or {'levels': 14, 'cmap': 'RdBu_r'}
        CS = ax.tricontourf(x, y, z, **kws)

        if colorbar:
            fig: plt.Figure = plt.gcf()
            fig.colorbar(CS, ax=ax)

    if show_points:
        kws = show_points_kws or {'color': 'k', 'marker': 'o', 'ms': 3, "linestyle": ""}
        ax.plot(x, y, **kws)

    process_axis(ax, **kwargs)

    if label_contours:
        kws = label_contour_kws or {"colors": "black"}
        ax.clabel(CS, **kws)

    if show:
        plt.show()

    return ax


def dumbbell_plot(
        start,
        end,
        labels=None,
        start_kws: dict = None,
        end_kws: dict = None,
        line_kws: dict = None,
        show: bool = True,
        ax: plt.Axes = None,
        **kwargs
) -> plt.Axes:
    """
    Dumbell plot which indicates variation of several variables
    from start to end.

    Parameters
    ----------
        start : list, array, series
            an array consisting of starting values
        end : list, array, series
            an array consisting of end values
        labels : list, array, series, optional
            names of values in start/end arrays. It is used to label
            ticklabcls on y-axis
        start_kws : dict, optional
            any additional keyword arguments for `axes.scatter`_ to modify start
            markers such as `color` etc
        end_kws : dict, optional
            any additional keyword arguments for `axes.scatter`_ to modify end
            markers such as `color` etc
        line_kws : dict, optional
            any additional keyword arguments for `lines.Line2D`_ to modify line
            style/color which connects dumbbells.
        show : bool, optional
            whether to show the plot or not
        **kwargs :
            any additional keyword arguments for `process_axis`.

    Returns
    -------
        matplotlib Axes object.

    Examples
    --------
        >>> import numpy as np
        >>> from easy_mpl import dumbbell_plot
        >>> st = np.random.randint(1, 5, 10)
        >>> en = np.random.randint(11, 20, 10)
        >>> dumbbell_plot(st, en)
         modify line color
        >>> dumbbell_plot(st, en, line_kws={'color':"black"})


    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html

    """
    if ax is None:
        ax = plt.gca()

    # convert starting and ending values to 1d array
    start = to_1d_array(start)
    end = to_1d_array(end)

    index = np.arange(len(start))

    assert len(start) == len(end) == len(index)

    if labels is None:
        labels = np.arange(len(index))

    line_kws = line_kws or {'color': 'skyblue'}
    # draw line segment
    def lien_segment(p1, p2, axes):
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], **line_kws)
        axes.add_line(l)
        return

    # assigning colors
    start_kws = start_kws or {'color': '#a3c4dc'}
    end_kws = end_kws or {'color': '#0e668b'}

    # plotting points for starting and ending values
    ax, _ = scatter(y=index, x=start, show=False, ax=ax, **start_kws)
    ax, _ = scatter(y=index, x=end, ax=ax, show=False, **end_kws)

    # joining points together using line segments
    for idx, _p1, _p2 in zip(index, end, start):
        lien_segment([_p1, idx], [_p2, idx], ax)

    # set labels
    ax.set_yticks(index)
    ax.set_yticklabels(labels)

    if kwargs:
        process_axis(ax=ax, **kwargs)
    # show plot if show=True
    if show:
        plt.show()

    return ax
