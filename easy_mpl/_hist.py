
__all__ = ["hist"]

from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from .utils import kde, deprecated_argument
from .utils import process_axes, is_dataframe, create_subplots, is_series

@deprecated_argument(x="data")
def hist(
        data: Union[list, np.ndarray, "Dataframe", "Series"],
        labels:Union[str, List[str]] = None,
        share_axes:bool = True,
        grid: bool = True,
        subplots_kws:dict = None,
        add_kde:bool = False,
        kde_kws:dict = None,
        line_kws:dict = None,
        ax: plt.Axes = None,
        ax_kws: dict = None,
        show: bool = True,
        return_axes:bool = False,
        **kwargs
):
    """
    one stop shop for histogram

    Parameters
    -----------
        data : list, array, optional
            array like, numpy ndarray or pandas DataFrame, or list of arrays
        labels : list/str optional
            names of the arrays, used for setting the legend
        share_axes : bool (default=True)
            whether to draw all the histograms on one axes or not?
        grid : bool, optional
            whether to show the grid or not
        subplots_kws : dict
            kws which go to plt.subplots() such as figure size (width, height)
        add_kde: bool, (default=False)
            whether to add a line representing kernel densitiy estimation or not
        kde_kws : dict
            keyword arguments to calculate kde. These will go to :func:`easy_mpl.utils.kde`
            function.
        line_kws : dict
            keyword arguments for drawing the kde line. These will go to
            :obj:`matplotlib.axes.Axes.plot` function
        ax : plt.Axes, optional
            axes on which to draw the plot
        ax_kws : dict
            keyword arguments for :py:func:`easy_mpl.utils.process_axes`
        show : bool, optional
            whether to show the plot or not
        return_axes : bool, (default=False)
            whether to return the axes objects on which histogram/histograms are
            drawn or not. If True, then the function returns two objects, the
            first is a tuple (output of axes.hist) and second is the axes/list of axes
            on which histogram is drawn
        **kwargs : optional
            any keyword arguments for :obj:`matplotlib.axes.Axes.hist`

    Returns
    -------
        same what is returned by :obj:`matplotlib.axes.Axes.hist`

    Example
    --------
        >>> from easy_mpl import hist
        >>> import numpy as np
        >>> hist(np.random.random((10, 1)))

    See :ref:`sphx_glr_auto_examples_hist.py` for more examples

    .. _axes.hist:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    """

    if isinstance(data, np.ndarray):
        if len(data) == data.size:
            X = [data]
            names = [None]
        else:
            X = [data[:, i] for i in range(data.shape[1])]
            names = [f"{i}" for i in range(data.shape[1])]

    elif is_dataframe(data):
        X = []
        for col in data.columns:
            X.append(data[col].values)
        names = data.columns.tolist()

    elif is_series(data):
        X = [data.values]
        names = [data.name]

    elif isinstance(data, (list, tuple)) and isinstance(data[0], (list, tuple, np.ndarray)):
        assert all([len(np.array(array_like))==np.array(array_like).size for array_like in data]), f"""
        All arrays must be one dimensional."""
        X = [np.array(x_).reshape(-1,) for x_ in data]
        names = [None]*len(X)

    elif isinstance(data, (list, tuple)) and is_dataframe(data[0]):
        X = [*data]
        names = [None]*len(X)

    elif isinstance(data, (list, tuple)) and not is_dataframe(data[0]):
        X = [data]
        names = [None]
    else:
        raise ValueError(f"unrecognized type of x {type(data)}")

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        assert len(labels) == len(names), f"{len(names)} does not match data"
        names = labels

    if share_axes:
        nplots = 1
    else:
        nplots = len(names)

    if nplots == 1:
        share_axes = True

    if subplots_kws is None:
        subplots_kws = dict()

    if not kde_kws:
        kde_kws = dict()
    _line_kws = {'color': 'k'}
    if line_kws:
        _line_kws.update(line_kws)

    f, axes = create_subplots(nplots, ax=ax, **subplots_kws)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    outs = []

    for idx, x, name in zip(range(len(names)), X, names):
        if name:
            kwargs['label'] = name

        if add_kde:
            kwargs['density'] = True

        if share_axes:
            assert isinstance(axes, plt.Axes)
            ax = axes
        else:
            ax = axes[idx]

        out = ax.hist(x, **kwargs)
        outs.append(out)

        if add_kde:
            kde_kws['bins'] = len(out[0])
            add_kde_line(ax, x, kde_kws, _line_kws)

        _ax_kws = dict(grid=grid)
        if ax_kws:
            _ax_kws.update(ax_kws)
        process_axes(ax, **_ax_kws)

        if name:
            ax.legend()

    if show:
        plt.show()

    if len(outs)==1:
        outs = outs[0]

    if return_axes:
        return outs, axes

    return outs


def add_kde_line(axes:plt.Axes, data, kde_kws:dict, line_kws:dict):
    ind, y = kde(data, **kde_kws)
    axes.plot(ind, y, **line_kws)
    return
