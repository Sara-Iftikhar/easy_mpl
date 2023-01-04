
__all__ = ["dumbbell_plot"]

from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .utils import is_rgb
from .utils import make_clrs_from_cmap
from .utils import to_1d_array, process_axes
from ._scatter import scatter


def dumbbell_plot(
        start,
        end,
        labels=None,
        line_color = None,
        start_marker_color = None,
        end_marker_color = None,
        start_kws: dict = None,
        end_kws: dict = None,
        line_kws: dict = None,
        sort_start:str = None,
        sort_end:str = None,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        show: bool = True
) -> Tuple[plt.Axes, mpl.collections.PathCollection, mpl.collections.PathCollection]:
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
        line_color :
            color for lines.  This can be a color name, rbg value, array of rbg values
            for each marker or a color palette name. This can be used to have separate
            color for a each line.
        start_marker_color :
            color for starting markers. This can be a color name, rbg value, array
            of rbg values for each marker or a color palette name. This can be
            used to have separate color for a each marker.
        end_marker_color :
            color for end markers. T This can be a color name, rbg value, array of
            rbg values for each marker or a color palette name. his can be used to
            have separate color for a each marker.
        start_kws : dict, optional
            any additional keyword arguments for :py:func:`easy_mpl.utils.scatter` to modify start
            markers such as ``color``, ``label`` etc
        end_kws : dict, optional
            any additional keyword arguments for :py:func:`easy_mpl.utils.scatter` to modify end
            markers such as ``color``, ``label`` etc
        line_kws : dict, optional
            any additional keyword arguments for `lines.Line2D`_ to modify line
            style/color which connects dumbbells.
        sort_start : str (default=None)
            either "ascend" or "descend"
        sort_end : str (default=None)
            either "ascend" or "descend"
        ax : plt.Axes, optional
            matplotlib axes object to work with. If not given then currently available
            axes will be used.
        ax_kws : dict optional
            any keyword arguments for :py:func:`easy_mpl.utils.process_axes`.
        show : bool, optional
            whether to show the plot or not

    Returns
    -------
    axes :
        :obj:`matplotlib.axes` matplotlib axes object on which dumbells are drawn
    st_pc :
        :obj:`matplotlib.collections.PathCollection`
    en_pc :
        :obj:`matplotlib.collections.PathCollection`

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

    line_colors = _get_color(line_color, _line_kws, len(start))

    # assigning colors
    _start_kws = {'color': '#a3c4dc', "label": "Start"}
    if start_kws:
        _start_kws.update(start_kws)

    _end_kws = {'color': '#0e668b', "label": "End"}
    if end_kws:
        _end_kws.update(end_kws)

    st_mc_colors = _get_color(start_marker_color, _start_kws, len(start))
    en_mc_colors = _get_color(end_marker_color, _end_kws, len(start))

    if sort_start:
        start, end, labels, line_colors, st_mc_colors, en_mc_colors = _handle_sort(
            sort_start, start, start, end, labels, line_colors, st_mc_colors, en_mc_colors)
    elif sort_end:
        start, end, labels, line_colors, st_mc_colors, en_mc_colors  = _handle_sort(
            sort_end, end, start, end, labels, line_colors, st_mc_colors, en_mc_colors)

    # draw line segment
    def line_segment(p1, p2, axes, color):
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, **_line_kws)
        axes.add_line(l)
        return

    # joining points together using line segments
    for (_idx, idx), _p1, _p2 in zip(enumerate(index), end, start):
        line_segment([_p1, idx], [_p2, idx], ax, color=line_colors[_idx])

    # circles are plotted after line so that lines don't enter inside the circles

    # plotting points for starting and ending values
    ax, st_paths = scatter(y=index, x=start, show=False, ax=ax,
                    color=st_mc_colors, **_start_kws)
    ax, en_paths = scatter(y=index, x=end, ax=ax, show=False,
                    color=en_mc_colors, **_end_kws)

    ax.legend()

    # set labels
    ax.set_yticks(index)
    ax.set_yticklabels(labels)

    if ax_kws:
        process_axes(ax=ax, **ax_kws)

    # show plot if show=True
    if show:
        plt.show()

    return ax, st_paths, en_paths


def _get_color(sugg_clr, kws, n)->list:

    if sugg_clr is None:
        colors = [kws['color'] for _ in range(n)]
    elif isinstance(sugg_clr, str):
        if sugg_clr in plt.colormaps():
            # todo
            #  this will result in wrong colorbar if these colors are used
            #  in for plot/scatter
            colors = make_clrs_from_cmap(sugg_clr, n, 0.1, 0.9)
        else:  # 'k'
            colors = [sugg_clr for _ in range(n)]
    elif is_rgb(sugg_clr):
        colors = [sugg_clr for _ in range(n)]
    else:
        assert hasattr(sugg_clr, '__len__') and len(sugg_clr) == n, f"Invalid color {sugg_clr}"
        colors = sugg_clr

    kws.pop('color')

    return colors


def _handle_sort(sort_type, sort_wrt, start, end, labels,
                 line_colors, st_mc_clr, en_mc_clr):

    assert sort_type in ["ascend", "descend"]
    if sort_type == "ascend":
        sort_idx = np.argsort(sort_wrt)
    else:
        sort_idx = np.flip(np.argsort(sort_wrt))

    start = np.array(start)[sort_idx]
    end = np.array(end)[sort_idx]
    labels = np.array(labels)[sort_idx]

    if not isinstance(line_colors, str):
        line_colors = np.array(line_colors)[sort_idx]

    if not isinstance(st_mc_clr, str):
        st_mc_clr = np.array(st_mc_clr)[sort_idx]

    if not isinstance(en_mc_clr, str):
        en_mc_clr = np.array(en_mc_clr)[sort_idx]

    return start, end, labels, line_colors, st_mc_clr, en_mc_clr