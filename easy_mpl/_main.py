__all__ = [
    "plot",
    "bar_chart",
    "regplot",
    "imshow",
    "hist",
    "pie",
    "scatter",
    "contour",
    "dumbbell_plot",
    "ridge",
    "parallel_coordinates",
    "lollipop_plot",
    "circular_bar_plot",
    "spider_plot",
]

import math
import random
from typing import Union, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.gridspec as grid_spec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import kde
from .utils import register_projections
from .utils import to_1d_array, make_cols_from_cmap, _regplot, process_axis
from .utils import BAR_CMAPS, regplot_combs, RIDGE_CMAPS, annotate_imshow


def bar_chart(
        values,
        labels=None,
        orient='h',
        sort=False,
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
        sort : bool, optional
            whether to sort the bars based upon their values or not
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
        matplotlib Axes

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
        if isinstance(values, (pd.DataFrame, pd.Series)):
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

    if orient in ['h', 'horizontal']:
        bar = ax.barh(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, rotation=rotation)

        if bar_labels is not None:
            bar_label_kws = bar_label_kws or {'label_type': 'center'}
            ax.bar_label(bar, labels=bar_labels, **bar_label_kws)

        if errors is not None:
            ax.errorbar(values, np.arange(len(values)), xerr=errors, fmt=".",
                        color="black")

    else:
        bar = ax.bar(np.arange(len(values)), values, color=color, **kwargs)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=rotation)

        if bar_labels is not None:
            bar_label_kws = bar_label_kws or {'label_type': 'center'}
            ax.bar_label(bar, labels=bar_labels, **bar_label_kws)

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


def plot(
        *args,
        show: bool = True,
        **kwargs
) -> plt.Axes:
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
                   'drawstyle', 'y_data', 'url', 'mfc', 'mec', 'mew', 'mfcalt', 'snap', 'markersize',
                   'lw', 'ls', 'ds', 'c', 'facecolor', 'markeredgecolor', 'markeredgewidth',
                   'markerfacecolor', 'markerfacesize', 'markerfacecoloralt',
                   )
    _plot_kwargs = {}
    for arg in plot_kwargs:
        if arg in kwargs:
            _plot_kwargs[arg] = kwargs.pop(arg)

    plot_args = []

    marker = None
    if len(args) == 1:
        data, = args
        data = [data]
    elif len(args) == 2 and not isinstance(args[1], str):
        data = args
    elif len(args) == 2 and isinstance(args[1], str):
        data, marker = args[0], args[1]
        data = [data]
    elif len(args) == 3:
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
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    s = data[0]
    if isinstance(s, pd.Series):
        kwargs['min_xticks'] = kwargs.get('min_xticks', 3)
        kwargs['max_xticks'] = kwargs.get('max_xticks', 5)
        kwargs['xlabel'] = kwargs.get('xlabel', s.index.name)
        kwargs['ylabel'] = kwargs.get('ylabel', s.name)
    elif isinstance(s, pd.DataFrame):
        kwargs['min_xticks'] = kwargs.get('min_xticks', 3)
        kwargs['max_xticks'] = kwargs.get('max_xticks', 5)
        if s.shape[1] == 1:
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


def imshow(
        values,
        xlabel=None,
        title=None,
        ylabel=None,
        yticklabels=None,
        xticklabels=None,
        show=True,
        annotate=False,
        annotate_kws:dict = None,
        colorbar: bool = False,
        ax=None,
        white_grid: bool = False,
        cb_tick_params: dict = None,
        **kwargs
) -> tuple:
    """
    One stop shop for matplotlib's imshow function

    Parameters
    ----------
        values: 2d array
            the image/data to show. It must bt 2 dimensional. It can also
            be dataframe.
        xlabel:  str, optional
        ylabel : str, optional
        title : str, optional
        show : bool, optional
            whether to show the plot or not
        annotate : bool, optional
            whether to annotate the heatmap or not
        annotate_kws : dict, optional
            a dictionary with following possible keys

                - ha : horizontal alighnment (default="center")
                - va : vertical alighnment (default="center")
                - fmt : format (default='%.2f')
                - textcolors : colors for axes.text
                - threshold : threshold to be used for annotation
                - **kws : any other keyword argument for axes.text

        colorbar : bool, optional
            whether to draw colorbar or not
        xticklabels : list, optional
            tick labels for x-axis. For DataFrames, column names are used by default.
        yticklabels :  list, optional
            tick labels for y-axis. For DataFrames, index is used by default
        ax : plt.Axes, optional
            if not given, current available axes will be used
        white_grid : bool, optional (default=False)
            whether to show the white grids or not. This will also turn off the spines.
        cb_tick_params : dict, optional
            tick params for colorbar. for example ``pad`` or ``orientation``
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
        ... # show colorbar
        >>> imshow(x, colorbar=True)
        ... # setting white grid lines and annotation
        >>> data = np.random.random((4, 10))
        >>> imshow(data, cmap="YlGn",
        ...        xticklabels=[f"Feature {i}" for i in range(data.shape[1])],
        ...        white_grid=True, annotate=True,
        ...        colorbar=True)
    .. _axes.imshow:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
    """

    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    if isinstance(values, pd.DataFrame):
        if not xticklabels:
            xticklabels = values.columns.to_list()
        if not yticklabels:
            yticklabels = values.index.tolist()
        # when data in dataframe is object type, it causes error in plotting
        # the best way to convert series in df to number is to use to_numeric
        values = np.column_stack([pd.to_numeric(values.iloc[:, i]) for i in range(values.shape[1])])

    tick_params = {}
    if 'ticks' in kwargs:
        tick_params['ticks'] = kwargs.pop('ticks')

    im = ax.imshow(values, **kwargs)

    if annotate_kws is None:
        annotate_kws = {}

    assert isinstance(annotate_kws, dict)

    _annotate_kws = {
        'ha':"center",
        "va": "center",
        "fmt": '%.2f',
        "textcolors": ("black", "white"),
        "threshold": None
    }

    _annotate_kws.update(annotate_kws)

    if annotate:
        annotate_imshow(im, values, **_annotate_kws)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        if len(xticklabels) > 5:
            ax.set_xticklabels(xticklabels, rotation=70)
        ax.set_xticklabels(xticklabels)

    process_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    if white_grid:
        # Turn spines off and create white grid.
        if isinstance(ax.spines, dict):
            for sp in ax.spines:
                ax.spines[sp].set_visible(False)
        else:
            ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(values.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(values.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    if colorbar:
        cb_tick_params = cb_tick_params or {'pad': 0.2, 'orientation': 'vertical'}
        # https://stackoverflow.com/a/18195921/5982232
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        fig: plt.Figure = plt.gcf()
        cb = fig.colorbar(im, cax=cax, **cb_tick_params)

    if show:
        plt.show()

    return ax, im


def hist(
        x: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        hist_kws: dict = None,
        grid: bool = True,
        ax: plt.Axes = None,
        show: bool = True,
        **kwargs
) -> plt.Axes:
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
        >>> import numpy as np
        >>> hist(np.random.random((10, 1)))

    .. _axes.hist:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    """

    if not ax:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

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
) -> plt.Axes:
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
        >>> import numpy as np
        >>> from easy_mpl import pie
        >>> pie(np.random.randint(0, 3, 100))
        or by directly providing fractions
        >>> pie([0.2, 0.3, 0.1, 0.4])
        ... # to explode 0.3
        >>> explode = (0, 0.1, 0, 0, 0)
        >>> pie(fractions=[0.2, 0.3, 0.15, 0.25, 0.1], explode=explode)

    .. _axes.pie:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
    """
    # todo, add example for explode and partial pie chart
    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

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
) -> Tuple[plt.Axes, mpl.collections.PathCollection]:
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
        >>> import matplotlib.pyplot as plt
        >>> x_ = np.random.random(100)
        >>> y_ = np.random.random(100)
        >>> scatter(x_, y_, show=False)
        ... # show colorbar
        >>> scatter(x_, y_, colorbar=True, show=False)
        ... # retrieve axes for further processing
        >>> axes, _ = scatter(x_, y_, show=False)
        >>> assert isinstance(axes, plt.Axes)

    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

    """
    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

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
        ... # show contour labels
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
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

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
            markers such as ``color``, ``label`` etc
        end_kws : dict, optional
            any additional keyword arguments for `axes.scatter`_ to modify end
            markers such as ``color``, ``label`` etc
        line_kws : dict, optional
            any additional keyword arguments for `lines.Line2D`_ to modify line
            style/color which connects dumbbells.
        show : bool, optional
            whether to show the plot or not
        ax : plt.Axes, optional
            matplotlib axes object to work with. If not given then currently available
            axes will be used.
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
        ... # modify line color
        >>> dumbbell_plot(st, en, line_kws={'color':"black"})


    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

    .. _lines.Line2D:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html

    """
    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

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
    start_kws = start_kws or {'color': '#a3c4dc', "label": "start"}
    end_kws = end_kws or {'color': '#0e668b', "label": "end"}

    # plotting points for starting and ending values
    ax, _ = scatter(y=index, x=start, show=False, ax=ax, **start_kws)
    ax, _ = scatter(y=index, x=end, ax=ax, show=False, **end_kws)

    ax.legend()

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
        plt.tight_layout()  # todo should we put it outside of if?
        plt.show()

    return ax


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


def parallel_coordinates(
        data: Union[pd.DataFrame, np.ndarray],
        categories: Union[np.ndarray, list] = None,
        names: list = None,
        cmap: str = None,
        linestyle: str = "bezier",
        names_fontsize: int = 14,
        title: str = 'Parallel Coordinates Plot',
        figsize: tuple = None,
        ticklabel_kws: dict = None,
        show: bool = True
) -> plt.Axes:
    """
    parallel coordinates plot
    modifying after https://stackoverflow.com/a/60401570/5982232

    Parameters
    ----------
        data : array, DataFrame
            a two dimensional array with the shape (rows, columns).
        categories : list, array
            1 dimensional array which contain class labels of the of each row in
            data. It can be either categorical or continuous numerical values.
            If not given, colorbar will not be drawn. The length of categroes
            array must be equal to length of/rows in data.
        names : list, optional
            Labels for columns in data. It's length should be equal to number of
            oclumns in data.
        cmap : str, optional
            colormap to be used
        names_fontsize : int, optional
            fontsize for names
        linestyle : str, optional
            either "straight" or "bezier". Default is "bezier".
        title : str, optional
            title for the Figure
        figsize : tuple, optional
            figure size
        ticklabel_kws : dict, optional
            keyword arguments for ticklabels on y-axis
        show : bool, optional
            whether to show the plot or not

    Returns
    -------
        matplotlib Axes

    Examples
    --------
    >>> import random
    >>> import numpy as np
    >>> import pandas as pd
    >>> from easy_mpl import parallel_coordinates
    ...
    >>> ynames = ['P1', 'P2', 'P3', 'P4', 'P5']  # feature/column names
    >>> N1, N2, N3 = 10, 5, 8
    >>> N = N1 + N2 + N3
    >>> categories_ = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> y1 = np.random.uniform(0, 10, N) + 7
    >>> y2 = np.sin(np.random.uniform(0, np.pi, N))
    >>> y3 = np.random.binomial(300, 1 / 10, N)
    >>> y4 = np.random.binomial(200, 1 / 3, N)
    >>> y5 = np.random.uniform(0, 800, N)
    ... # combine all arrays into a pandas DataFrame
    >>> data_np = np.column_stack((y1, y2, y3, y4, y5))
    >>> data_df = pd.DataFrame(data_np, columns=ynames)
    ... # using a DataFrame to draw parallel coordinates
    >>> parallel_coordinates(data_df, names=ynames)
    ... # using continuous values for categories
    >>> parallel_coordinates(data_df, names=ynames, categories=np.random.randint(0, 5, N))
    ... # using categorical classes
    >>> parallel_coordinates(data_df, names=ynames, categories=random.choices(categories_, k=N))
    ... # using numpy array instead of DataFrame
    >>> parallel_coordinates(data_df.values, names=ynames)
    ... # with customized tick labels
    >>> parallel_coordinates(data_df.values, ticklabel_kws={"fontsize": 8, "color": "red"})
    ... # using straight lines instead of bezier
    >>> parallel_coordinates(data_df, linestyle="straight")
    ... # with categorical class labels
    >>> data_df['P5'] = random.choices(categories_, k=N)
    >>> parallel_coordinates(data_df, names=ynames)
    ... # with categorical class labels and customized ticklabels
    >>> data_df['P5'] = random.choices(categories_, k=N)
    >>> parallel_coordinates(data_df,  ticklabel_kws={"fontsize": 8, "color": "red"})

    Note
    ----
        If nans are present in data or categories, all the corresponding enteries/rows
        will be removed.
    """

    if cmap is None:
        cmap = "Blues"

    if isinstance(data, np.ndarray):
        assert data.ndim == 2, f"{data.ndim} dimensional data not allowed. It must be 2d"
        if names is None:
            names = [f"Feat_{i}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=names)

    if isinstance(data, pd.DataFrame):
        names = names or data.columns.tolist()

    if len(names) != data.shape[1]:
        raise ValueError(f"""
            provided names have length {len(names)} but data has {data.shape[1]} columns""")

    show_colorbar = True
    if categories is None:
        show_colorbar = False
        categories = np.linspace(0, 1, len(data))

    categories = np.array(categories)
    assert len(categories) == len(data)

    # remove NaN values based upon nan values in data
    if data.isna().sum().sum() > 0:
        df_nan_idx = data.isna().any(axis=1)
        categories = categories[~df_nan_idx]
        data = data[~df_nan_idx]

    _is_categorical = False
    cat_encoded = categories
    if not np.issubdtype(categories.dtype, np.number):
        # category contains categorical/non-numeri values
        cat_encoded = label_encoder(categories)
        _is_categorical = True

    if not _is_categorical:  # because we can't do np.isnan for categorical values
        # if there are still any nans in categories, remove them
        cat_nan_idx = np.isnan(categories)
        if cat_nan_idx.any():
            categories = categories[~cat_nan_idx]
            data = data[~cat_nan_idx]

    num_cols = data.shape[1]
    num_lines = len(data)

    # find out which columns are categorical and which are numerical
    enc_data = data.copy()
    cols = {}
    for idx, col in enumerate(data.columns):
        _col = data[col]
        if is_categorical(data[col].values):
            col_encoded = label_encoder(data[col].values)
            cols[idx] = {'cat': True, 'original': _col}
            enc_data[col] = col_encoded
        else:
            cols[idx] = {'cat': False}

    # organize the data
    enc_data = enc_data.astype(float)
    ymins = np.min(enc_data.values, axis=0)  # ys.min(axis=0)
    ymaxs = np.max(enc_data.values, axis=0)  # ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(enc_data.values)
    zs[:, 0] = enc_data.iloc[:, 0]
    zs[:, 1:] = (enc_data.iloc[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=figsize)

    axes = [host] + [host.twinx() for _ in range(num_cols - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (num_cols - 1)))

        if cols[i]['cat']:
            categories = np.unique(cols[i]['original'])
            new_ticks = np.unique(enc_data.iloc[:, i]).astype("float32")
            ax.set_yticks(new_ticks)
            ax.set_yticklabels(categories)

        if ticklabel_kws:
            if cols[i]['cat']:
                ticks_loc = [l._text for l in ax.get_yticklabels()]
            else:
                ticks_loc = ax.get_yticks().tolist()

            ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels([label_format(x) for x in ticks_loc], **ticklabel_kws)

    host.set_xlim(0, num_cols - 1)
    host.set_xticks(range(num_cols))
    host.set_xticklabels(names, fontsize=names_fontsize)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    if title:
        host.set_title(title, fontsize=18)

    # category between 0.2,1 to map colors to their values
    cat_norm = _rescale(cat_encoded, 0.2)

    for j in range(num_lines):
        # color of each line is based upon corresponding value in category
        colors = getattr(cm, cmap)(cat_norm[j])

        if linestyle == "straight":
            # to just draw straight lines between the axes:
            host.plot(range(num_cols), zs[j, :], c=colors)
        else:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            x_coords = [x for x in np.linspace(0, len(data) - 1, len(data) * 3 - 2, endpoint=True)]
            y_coords = np.repeat(zs[j, :], 3)[1:-1]
            verts = list(zip(x_coords, y_coords))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors)
            host.add_patch(patch)

    if show_colorbar:
        norm = cm.colors.Normalize(np.min(cat_encoded), np.max(cat_encoded))

        cb = cm.ScalarMappable(norm, cmap=cmap)

        if _is_categorical:
            cbar = fig.colorbar(cb, orientation="vertical", pad=0.1)
            ticks = cbar.get_ticks()
            new_ticks = np.linspace(ticks[0], ticks[-1], len(np.unique(categories)))
            cbar.set_ticks(new_ticks)
            cbar.set_ticklabels(np.unique(categories))

        else:
            cbar = fig.colorbar(cb, orientation="vertical", pad=0.1)

    plt.tight_layout()

    if show:
        plt.show()

    return host


def label_format(x):
    if isinstance(x, float):
        return round(x, 3)
    else:
        return x


def is_categorical(array) -> bool:
    return not np.issubdtype(array.dtype, np.number)


def _rescale(y, _min=0.0, _max=1.0):
    y_std = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

    return y_std * (_max - _min) + _min


def label_encoder(arr):
    # label encoder of numpy array with categorical values
    return np.unique(arr, return_inverse=True)[1]


def lollipop_plot(
        y, x=None,
        orientation: str = "vertical",
        sort: bool = False,
        line_style: str = '-',
        line_color: str = 'cyan',
        line_width: int = 1,
        line_kws: dict = None,
        marker_style: str = 'o',
        marker_color: str = 'teal',
        marker_size: int = 30,
        marker_kws: dict = None,
        show: bool = True,
        ax: plt.Axes = None,
        **kwargs) -> plt.Axes:
    """
    Plot a lollipop plot.

    Parameters
    ----------
    y : array_like, shape (n,), optional
        The y-coordinates of the data points.
    x : array_like, shape (n,)
        The x-coordinates of the data points.
    orientation : str, optional
        The orientation of the lollipops. Either "vertical" or "horizontal".
    sort : bool, optional
        Whether to sort the data points by their values or not.
        Only valid if `x` is not specified.
    line_style : str, optional
        The line style of the data points.
    line_color : str, optional
        The line color of the data points.
    line_width : float, optional
        The line width of the data points.
    line_kws : dict, optional
        The keyword arguments for the line. These arguments are passed to
        `matplotlib.axes.Axes.plot`_.
    marker_style : str, optional
        The marker style of the data points.
    marker_color : str, optional
        The marker color of the data points.
    marker_size : float, optional
        The marker size of the data points.
    marker_kws : dict, optional
        The keyword arguments for the marker. These arguments are passed to
        `matplotlib.axes.Axes.scatter`_.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not given, current available axes will be used.
    **kwargs : optional
        Additional keyword arguments to pass to the process_axis function.

    Returns
    -------
    plt.Axes :
        The axes on which the plot was drawn.

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl import lollipop_plot
    >>> y = np.random.randint(0, 10, size=10)
    ... # vanilla lollipop plot
    >>> lollipop_plot(y, title="vanilla")
    ... # use both x and y
    >>> lollipop_plot(y, np.linspace(0, 100, len(y)), title="with x and y")
    ... # use custom line style
    >>> lollipop_plot(y, line_style='--', title="with custom linestyle")
    ... # use custom marker style
    >>> lollipop_plot(y, marker_style='D', title="with custom marker style")
    ... # sort the data points before plotting
    >>> lollipop_plot(y, sort=True, title="sort")
    ... # horzontal orientation of lollipops
    >>> y = np.random.randint(0, 20, size=10)
    >>> lollipop_plot(y, orientation="horizontal", title="horizontal")

    .. _matplotlib.axes.Axes.plot:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html

    .. _matplotlib.axes.Axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    """

    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    y = to_1d_array(y)

    if sort:
        idx = np.argsort(y)
        y = y[idx]
        assert x is None

    if x is None:
        x = np.arange(len(y))

    x = to_1d_array(x)

    marker_kws = marker_kws or {}
    line_kws = line_kws or {}

    if orientation == "vertical":
        _lollipop_vertical(ax, x, y, line_style, line_color, line_width, line_kws,
                           marker_style, marker_color, marker_size, marker_kws)
    else:
        _lollipop_horizontal(ax, x, y, line_style, line_color, line_width, line_kws,
                             marker_style, marker_color, marker_size, marker_kws)

    process_axis(ax=ax, **kwargs)

    if show:
        plt.show()

    return ax


def _lollipop_vertical(ax, x, y, line_style, line_color, line_width, line_kws,
                       marker_style, marker_color, marker_size, marker_kws):
    ax.scatter(x, y, marker=marker_style, color=marker_color,
               s=marker_size, **marker_kws)
    ax.vlines(x, np.zeros(len(x)), y, color=line_color,
              linestyle=line_style, linewidth=line_width, **line_kws)
    return ax


def _lollipop_horizontal(ax, x, y, line_style, line_color, line_width, line_kws,
                         marker_style, marker_color, marker_size, marker_kws):
    ax.scatter(y, x, marker=marker_style, color=marker_color,
               s=marker_size, **marker_kws)
    ax.hlines(x, np.zeros(len(y)), y, color=line_color,
              linestyle=line_style, linewidth=line_width, **line_kws)
    return ax


def circular_bar_plot(
        data,
        labels: list = None,
        sort=False,
        color: Union[str, list, np.ndarray] = None,
        label_format: str = None,
        min_max_range: tuple = None,
        label_padding: int = 4,
        figsize: tuple = None,
        show: bool = True,
        text_kws: dict = None,
        **kwargs
) -> plt.Axes:
    """
    Plot a circular bar plot.

    Parameters
    ----------
    data : list, np.ndarray, pd.Series, dict
        Data to plot. If it is a dictionary, then its keys will be used
        as labels and values will be used as data.
    labels : list, optional
        Labels for each data point.
    sort : bool, optional
        Sort the data by the values.
    color : str, list, np.ndarray, optional
        Color for each data point. It can be a single color or a colormap from
        plt.colormaps.
    label_format : str, optional
        Format for the labels.
    min_max_range : tuple, optional
        Minimum and maximum range for normalizing the data.
    label_padding : int, optional
        space between the labels and the bars.
    figsize : tuple, optional
        Size of the figure.
    show : bool, optional
        Show the plot.
    **kwargs : optional
        Additional keyword arguments to pass to the process_axis function.

    Returns
    -------
    ax : plt.Axes
        Axes of the plot.

    Note
    ----
        If nan values are present in the data, they will be ignored.

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl import circular_bar_plot
    >>> data = np.random.random(50, )
    ... # basic
    >>> circular_bar_plot(data)
    ... # with names
    >>> names = [f"{i}" for i in range(50)]
    >>> circular_bar_plot(data, names)
    ... # sort values
    >>> circular_bar_plot(data, names, sort=True)
    ... # custom color map
    >>> circular_bar_plot(data, names, color='viridis')
    ... # custom min and max range
    >>> circular_bar_plot(data, names, min_max_range=(1, 10), label_padding=1)
    ... # custom label format
    >>> circular_bar_plot(data, names, label_format='{} {:.4f}')

    """

    text_kws = text_kws or {}

    plt.close('all')
    plt.figure(figsize=figsize or (8, 12))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    if isinstance(data, pd.DataFrame):
        values = data.values
    elif isinstance(data, dict):
        values = np.array(list(data.values()))
        labels = labels or list(data.keys())
    else:
        data = to_1d_array(data)
        values = data

    if labels is None:
        labels = ['' for _ in range(len(values))]
        label_format = label_format or "{} {:.2f}"
    else:
        label_format = label_format or "{}: {:.2f}"

    # remove nan values
    val_nan_idx = np.isnan(values)
    if val_nan_idx.any():
        values = values[~val_nan_idx]
        labels = [labels[i] for i in range(len(labels)) if not val_nan_idx[i]]

    if color is None:
        color = make_cols_from_cmap(random.choice(BAR_CMAPS), len(values), 0.2)
    elif isinstance(color, str) and color in plt.colormaps():
        color = make_cols_from_cmap(color, len(values), 0.2)
    else:
        color = color

    assert len(values) == len(labels)

    min_max_range = min_max_range or (30, 100)
    lower_limit = min_max_range[0]
    heights = _rescale(values.reshape(-1, 1), lower_limit, min_max_range[1]).reshape(-1, )

    if sort:
        sort_idx = np.argsort(heights)
        heights = heights[sort_idx]
        labels = [labels[i] for i in sort_idx]
        values = values[sort_idx]
        # color = color[sort_idx]

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2 * np.pi / len(heights)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(heights) + 1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=heights,
        width=width,
        bottom=lower_limit,
        linewidth=2,
        edgecolor="white",
        color=color,
    )

    # Add labels
    for bar, angle, label, val in zip(bars, angles, labels, values):

        label = label_format.format(label, val)

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle,
            y=lower_limit + bar.get_height() + label_padding,
            s=label,
            ha=alignment,
            va='center',
            rotation=rotation,
            rotation_mode="anchor",
            **text_kws
        )

    if kwargs:
        process_axis(ax, **kwargs)

    if show:
        plt.show()

    return ax


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
        ax : plt.Axes
            matplotlib axes on which to draw the axes
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
    >>> values = [-0.2, 0.1, 0.0, 0.1, 0.2, 0.3]
    >>> spider_plot(values=values)
    ... # specifying labels
    labels = ['a', 'b','c', 'd', 'e', 'f']
    >>> spider_plot(values=values, labels=labels)
    ... # specifying tick size
    >>> spider_plot(values, labels, xtick_kws={'size': 13})
    ...
    >>> df = pd.DataFrame.from_dict(
    ... {'summer': {'a': -0.2, 'b': 0.1, 'c': 0.0, 'd': 0.1, 'e': 0.2, 'f': 0.3},
    ... 'winter': {'a': -0.3, 'b': 0.1, 'c': 0.0, 'd': 0.2, 'e': 0.15,'f': 0.25}})
    >>> spider_plot(df, xtick_kws={'size': 13})
    ... # use polygon frame
    >>> spider_plot(values=values, frame="polygon")

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

    _xtick_kws = {'color': 'grey', 'size': 8}
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

    plt.gca().set_rmax(.4)

    _leg_kws = {'loc': 'best', 'labelspacing': 0.1, 'fontsize': 'small'}
    leg_kws = leg_kws or {}
    _leg_kws.update(leg_kws)
    legend = ax.legend(labels, **_leg_kws)

    if show:
        plt.show()

    return ax
