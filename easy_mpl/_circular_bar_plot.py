
__all__ = ["circular_bar_plot"]

import random
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import _rescale
from .utils import BAR_CMAPS
from .utils import to_1d_array, make_cols_from_cmap, process_axes


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
    show : bool, optional (default=True)
        Show the plot.
    text_kws : dict, optional (default=None)
        keyword arguments for axes.text()
    **kwargs : optional
        Additional keyword arguments to pass to the process_axis function.

    Returns
    -------
    ax : plt.Axes
        Axes of the plot.

    See :ref:`sphx_glr_auto_examples_circular_bar_plot.py` for more examples

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

    if hasattr(data, "values") and hasattr(data, "columns"):
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
        process_axes(ax, **kwargs)

    if show:
        plt.show()

    return ax
