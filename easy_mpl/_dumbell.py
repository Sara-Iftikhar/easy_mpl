
__all__ = ["dumbbell_plot"]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .utils import to_1d_array, process_axes
from ._scatter import scatter

def dumbbell_plot(
        start,
        end,
        labels=None,
        start_kws: dict = None,
        end_kws: dict = None,
        line_kws: dict = None,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        show: bool = True,
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
            any additional keyword arguments for :py:func:`easy_mpl.utils.scatter` to modify start
            markers such as ``color``, ``label`` etc
        end_kws : dict, optional
            any additional keyword arguments for :py:func:`easy_mpl.utils.scatter` to modify end
            markers such as ``color``, ``label`` etc
        line_kws : dict, optional
            any additional keyword arguments for `lines.Line2D`_ to modify line
            style/color which connects dumbbells.
        ax : plt.Axes, optional
            matplotlib axes object to work with. If not given then currently available
            axes will be used.
        ax_kws : dict optional
            any keyword arguments for :py:func:`easy_mpl.utils.process_axes`.
        show : bool, optional
            whether to show the plot or not
        **kwargs :
            any additional keyword arguments for

    Returns
    -------
    :obj:`matplotlib.axes`
        matplotlib axes object on which dumbells are drawn

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

    .. _lines.Line2D:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html

    """

    if ax_kws is None:
        ax_kws = dict()

    if ax is None:
        ax = plt.gca()
        if 'figsize' in ax_kws:
            figsize = ax_kws.pop('figsize')
            ax.figure.set_size_inches(figsize)

    # convert starting and ending values to 1d array
    start = to_1d_array(start)
    end = to_1d_array(end)

    index = np.arange(len(start))

    assert len(start) == len(end) == len(index)

    if labels is None:
        labels = np.arange(len(index))

    _line_kws = {'color': 'skyblue'}
    if line_kws is not None:
        _line_kws.update(line_kws)

    # draw line segment
    def lien_segment(p1, p2, axes):
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], **_line_kws)
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

    if ax_kws:
        process_axes(ax=ax, **ax_kws)

    # show plot if show=True
    if show:
        plt.tight_layout()  # todo should we put it outside of if?
        plt.show()

    return ax
