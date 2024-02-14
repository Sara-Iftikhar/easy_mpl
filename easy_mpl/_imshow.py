

__all__ = ["imshow"]

from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import despine_axes
from .utils import add_cbar
from .utils import process_axes
from .utils import annotate_imshow


def imshow(
        values,
        yticklabels=None,
        xticklabels=None,
        annotate:bool = False,
        annotate_kws:dict = None,
        colorbar: bool = False,
        grid_params: dict = None,
        mask : Union[bool, str, np.ndarray] = None,
        cbar_params: dict = None,
        ax:plt.Axes = None,
        ax_kws: dict = None,
        show:bool = True,
        **kwargs
):
    """
    One stop shop for matplotlib's imshow function

    Parameters
    ----------
        values: 2d array
            the image/data to show. It must bt 2 dimensional. It can also
            be dataframe.
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
        grid_params : dict, optional (default=None)
            parameters to process grid. Allowed keys in the dictionary are following
                - ``border``, bool
                - ``linestyle``
                - ``linewidth``
                - ``color``
        mask :
            This argument can be used to hide part of heatmap from being displayed.
                - True : will only show the lower half
                - ``upper`` will only show the lower half
                - ``lower`` will only show the upper half
        cbar_params : dict, optional
            parameters that will go to :py:func`easy_mpl.utils.process_cbar` for colorbar.
            For example ``pad`` or ``orientation``
        ax : plt.Axes, optional
            if not given, current available axes will be used
        ax_kws : dict, optional (default=None)
            any keyword arguments for :py:func:`easy_mpl.utils.process_axes` function as dictionary
        show : bool, optional
            whether to show the plot or not
        **kwargs : optional
            any further keyword arguments for :obj:`matplotlib.axes.Axes.imshow`

    Returns
    -------
    matplotlib.image.AxesImage
        a :obj:`matplotlib.image.AxesImage`

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
        ...        grid_params={'border': True, 'color': 'w', 'linewidth': 2}, annotate=True,
        ...        colorbar=True)

    See :ref:`sphx_glr_auto_examples_imshow.py` for more examples

    """
    if ax_kws is None:
        ax_kws = dict()

    if ax is None:
        ax = plt.gca()
        if 'figsize' in ax_kws:
            figsize = ax_kws.pop('figsize')
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

    to_keep = None
    if mask is not None:
        if isinstance(mask, (str, bool)):
            _mask = np.tri(values.shape[0], k=-1)
            if mask == "lower":
                values = np.ma.array(values, mask=_mask)  # mask out the lower triangle
                to_keep = ['right', 'top']
            else:
                values = np.ma.array(values, mask=_mask).T
                to_keep = ['left', 'bottom']

    tick_params = {}
    if 'ticks' in kwargs:
        tick_params['ticks'] = kwargs.pop('ticks')

    im = ax.imshow(values, **kwargs)

    if to_keep:
        despine_axes(ax, keep=to_keep)


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

    if ax_kws:
        process_axes(ax, **ax_kws)

    if grid_params:
        process_grid(ax, values, **grid_params)

    if colorbar:
        if cbar_params is None:
            cbar_params = {}
        add_cbar(ax, im, **cbar_params)
        # cb_tick_params = cb_tick_params or {'pad': 0.2, 'orientation': 'vertical'}
        # # https://stackoverflow.com/a/18195921/5982232
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.2)
        # fig: plt.Figure = plt.gcf()
        # cb = fig.colorbar(im, cax=cax, **cb_tick_params)

    if show:
        plt.show()

    return im


def process_grid(
        ax:plt.Axes,
        data:np.ndarray,
        border:bool = False,
        color:str = "w",
        linewidth:Union[int, float] = 3,
        linestyle:str = '-'
):

    if not border:
        # Turn spines off and create white grid.
        if isinstance(ax.spines, dict):
            for sp in ax.spines:
                ax.spines[sp].set_visible(False)
        else:
            ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color=color, linestyle=linestyle, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)
    return