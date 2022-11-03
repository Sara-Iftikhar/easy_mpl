
__all__ = [
    "plot",
    "scatter",
    "dumbbell_plot"
]

from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .utils import to_1d_array, process_axis


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
        matplotlib Axes on which the plot is drawn. If ``show`` is False, this axes
        can be used for further processing

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

    See :ref:`sphx_glr_auto_examples_plot.py` for more examples

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

    if ax is None:
        # it is possible that as is given in kwargs but given as None
        ax = plt.gca()

    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
        ax.figure.set_size_inches(figsize)

    s = data[0]
    if hasattr(s, "index") and hasattr(s, "name"):
        kwargs['min_xticks'] = kwargs.get('min_xticks', 3)
        kwargs['max_xticks'] = kwargs.get('max_xticks', 5)
        kwargs['xlabel'] = kwargs.get('xlabel', s.index.name)
        kwargs['ylabel'] = kwargs.get('ylabel', s.name)
    elif hasattr(s, "values") and hasattr(s, "index") and hasattr(s, "columns"):
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
    tuple :
        A tuple whose first member is matplotlib Axes and second member is
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

    See :ref:`sphx_glr_auto_examples_scatter.py` for more examples

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

    See :ref:`sphx_glr_auto_examples_dumbell.py` for more examples

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
