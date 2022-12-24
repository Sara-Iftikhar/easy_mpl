

__all__ = ["imshow"]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import process_axes
from .utils import annotate_imshow


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
        ax_kws: dict = None,
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
        ax_kws : dict, optional (default=None)
            any keyword arguments for :py:func:`easy_mpl.utils.process_axes` function as dictionary
        **kwargs : optional
            any further keyword arguments for :obj:`matplotlib.axes.Axes.imshow`

    Returns
    -------
    tuple
        a tuple whose first vlaue is :obj:`matplotlib.axes` and second argument is :obj:`matplotlib.image.AxesImage`

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

    See :ref:`sphx_glr_auto_examples_imshow.py` for more examples

    """

    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    if hasattr(values, "values") and hasattr(values, "columns"):
        import pandas as pd  # don't make whole project dependent upon pandas

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

    if not ax_kws:
        ax_kws = dict()
    process_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, **ax_kws)

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
