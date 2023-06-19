
__all__ = ["scatter"]

from typing import Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .utils import to_1d_array, process_axes


def scatter(
        x,
        y,
        colorbar: bool = False,
        colorbar_orientation: str = "vertical",
        marker_labels: Union[list, np.ndarray] = None,
        text_kws : dict = None,
        xoffset = 0.1,
        yoffset = 0.1,
        ax: plt.Axes = None,
        ax_kws:dict = None,
        show: bool = True,
        **kwargs
) -> Tuple[plt.Axes, mpl.collections.PathCollection]:
    """
    scatter plot between two arrays x and y

    Parameters
    ----------
    x : list, array
        data for x-axis
    y : list, array
        data for y-axis
    colorbar : bool, optional
        whether to show the color bar or not
    colorbar_orientation : str, optional
        orientation of colorbar. Only relevant if ``colorbar`` is True
    marker_labels : list, array
        labels to annotate each marker. If given, each value must
        correspond to respective values in x,y arrays
    text_kws : dict
        only relevant if ``marker_labels`` are provided.
    xoffset : float
    yoffset : float
    ax : plt.Axes, optional
        :obj:`matplotlib.axes`, if not given, current available axes will be used
    ax_kws : dict (default=None)
        any keyword arguments for processing of axes which will
        be forwarded to :func:`easy_mpl.utils.prcess_axis`
    show : bool, optional (default=True)
        whether to show the plot or not
    **kwargs : optional
        any additional keyword arguments for :obj:`matplotlib.axes.Axes.scatter`

    Returns
    --------
    tuple :
        A tuple whose first member is :obj:`matplotlib.axes` and second member is
        :obj:`matplotlib.collections.PathCollection`

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

    """
    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    x = to_1d_array(x)
    y = to_1d_array(y)

    #if colorbar:
    #    if 'c' not in kwargs and 'color' not in kwargs:
    #        kwargs['c'] = np.arange(len(x))

    sc = ax.scatter(x, y, **kwargs)

    if marker_labels is not None:
        y = y.reshape(-1,)
        _text_kws = {}
        if text_kws is not None:
            _text_kws.update(text_kws)
        for i, txt in enumerate(marker_labels):
            ax.annotate(txt, (x[i] + xoffset, y[i] + yoffset), **_text_kws)

    if colorbar:
        fig: plt.Figure = ax.get_figure()
        fig.colorbar(sc, ax=ax, orientation=colorbar_orientation, pad=0.1)

    if ax_kws:
        process_axes(ax=ax, **ax_kws)

    if show:
        plt.show()

    return ax, sc
