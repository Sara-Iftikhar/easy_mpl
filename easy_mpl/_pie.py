

__all__ = ["pie"]

from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl.utils import process_axes


def pie(
        vals: Union[list, np.ndarray] = None,
        fractions: Union[list, np.ndarray] = None,
        labels: list = None,
        autopct = '%1.1f%%',
        ax: plt.Axes = None,
        ax_kws: dict = None,
        show: bool = True,
        **kwargs
) -> tuple:
    """
    draws the pie chart

    Parameters
    ----------
        vals : array like,
            unique values and their counts will be inferred from this array.
        fractions : list, array, optional
            if given, vals must not be given
        labels : list, array, optional
            labels for unique values in vals, if given, must be equal to unique vals
            in vals. Otherwise "unique_value (counts)" will be used for labeling.
        autopct : str (default='%1.1f%%')
            string defining method to represent percentage. Set this to
            None to not use this argument.
        ax : plt.Axes, optional
            the :obj:`matplotlib.axes` on which to draw, if not given current active axes will be used
        ax_kws : dict, optional
            keyword arguments for :py:func:`easy_mpl.utils.process_axes`
        show: bool, optional (default=True)
            whether to show the plot or not
        **kwargs: optional
            any keyword argument will go to :obj:`matplotlib.axes.Axes.pie`

    Returns
    -------
    outs
        same what is returned by :obj:`matplotlib.axes.Axes.pie`

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

    See :ref:`sphx_glr_auto_examples_pie.py` for more examples

    .. _axes.pie:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
    """
    # todo, add example for and partial pie chart

    if ax_kws is None:
        ax_kws = dict()

    if ax is None:
        ax = plt.gca()
        if 'figsize' in ax_kws:
            figsize = ax_kws.pop('figsize')
            ax.figure.set_size_inches(figsize)

    if fractions is None:
        uniques, counts = np.unique(vals, return_counts=True)
        fractions = counts / counts.sum()
        vals = {k:v for k,v in zip(uniques, counts)}

        if labels is None:
            labels = [f"{value} ({count}) " for value, count in vals.items()]
    else:
        assert vals is None

    outs = ax.pie(fractions,
                  labels=labels,
                  autopct=autopct,
           **kwargs)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if ax_kws:
        process_axes(ax, **ax_kws)

    if show:
        plt.show()

    return outs
