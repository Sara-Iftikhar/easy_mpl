
__all__ = ["plot"]

import numpy as np
import matplotlib.pyplot as plt

from .utils import process_axes, is_dataframe, is_series

# todo add share_axes argument

def plot(
        *args,
        show: bool = True,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        **kwargs
) -> plt.Axes:
    """
    One liner plot function. It's use is not more complex than :obj:`matplotlib.axes.Axes.plot` or
    :obj:`matplotlib.pyplot.plot` . However it accomplishes all in one line what requires multiple
    lines in matplotlib. args and kwargs can be anything which goes into :obj:`matplotlib.pyplot.plot`
    or :obj:`matplotlib.axes.Axes.plot`.

    Parameters
    ----------
        *args :
            either a single array or x and y arrays or anything which can go to
            :obj:`matplotlib.axes.Axes.plot` or anything which can got to :obj:`matplotlib.pyplot.plot` .
        ax : :obj:`matplotlib.axes`
            matplotlib axes object on which plot is to be drawn. If not given,
            then current active axes will be used.
        ax_kws : dict
            keyword arguments for :func:`easy_mpl.utils.process_axes`
        show : bool, optional (default=True)
            If set to True, plt.show() is called.
        **kwargs : optional
            Any keyword argument for :obj:`matplotlib.pyplot.plot` or :obj:`matplotlib.axes.Axes.plot`

    Returns
    -------
    :obj:`matplotlib.axes`
        :obj:`matplotlib.axes` on which the plot is drawn. If ``show`` is False, this axes
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
        >>> plot(np.random.random(100), '--*', ax_kws={'logy':True}, label='label')

    See :ref:`sphx_glr_auto_examples_plot.py` for more examples

    """

    if ax_kws is None:
        ax_kws = {}

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
        assert 'marker' not in kwargs  # we have alreay got marker

    if not ax:
        ax = plt.gca()

    if 'figsize' in ax_kws:
        figsize = ax_kws.pop('figsize')
        ax.figure.set_size_inches(figsize)

    s = data[0]
    if is_series(s):
        ax_kws['min_xticks'] = ax_kws.get('min_xticks', 3)
        ax_kws['max_xticks'] = ax_kws.get('max_xticks', 5)
        ax_kws['xlabel'] = ax_kws.get('xlabel', s.index.name)
        ax_kws['ylabel'] = ax_kws.get('ylabel', s.name)
    elif is_dataframe(s):
        ax_kws['min_xticks'] = ax_kws.get('min_xticks', 3)
        ax_kws['max_xticks'] = ax_kws.get('max_xticks', 5)
        if s.shape[1] == 1:
            ax_kws['xlabel'] = ax_kws.get('xlabel', s.index.name)
            ax_kws['ylabel'] = ax_kws.get('ylabel', s.columns.tolist()[0])
        else:
            ax_kws['xlabel'] = ax_kws.get('xlabel', s.index.name)
            ax_kws['label'] = ax_kws.get('label', s.columns.tolist())
            for col in s.columns:
                kwargs['label'] = col
                data[0] = s[col]
                ax.plot(*data, *plot_args, **kwargs)
            return _process_axis(ax, show, ax_kws)

    if 'label' in kwargs:
        ax_kws['label'] = kwargs['label']

    ax.plot(*data, *plot_args, **kwargs)
    return _process_axis(ax, show, ax_kws)


def _process_axis(ax, show, kwargs):
    if kwargs:
        ax = process_axes(ax=ax, **kwargs)
    if kwargs.get('save', False):
        plt.savefig(f"{kwargs.get('name', 'fig.png')}")
    if show:
        plt.show()
    return ax

