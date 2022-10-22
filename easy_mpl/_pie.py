

__all__ = ["pie"]

from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pie(
        vals: Union[list, np.ndarray] = None,
        fractions: Union[list, np.ndarray] = None,
        labels: list = None,
        ax: plt.Axes = None,
        title: str = None,
        show: bool = True,
        **kwargs
) -> plt.Axes:
    """
    pie chart

    Parameters
    ----------
        vals : array like,
            unique values and their counts will be inferred from this array.
        fractions : list, array, optional
            if given, vals must not be given
        labels : list, array, optional
            labels for unique values in vals, if given, must be equal to unique vals
            in vals. Otherwise "unique_value (counts)" will be used for labeling.
        ax : plt.Axes, optional
            the axes on which to draw, if not given current active axes will be used
        title: str, optional
            if given, will be used for title
        show: bool, optional
        **kwargs: optional
            any keyword argument will go to `axes.pie`_

    Returns
    -------
        a matplotlib axes. This can be used for further processing by making show=False.

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

    .. _axes.pie:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
    """
    # todo, add example for and partial pie chart
    if ax is None:
        ax = plt.gca()
        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
            ax.figure.set_size_inches(figsize)

    if fractions is None:
        fractions = pd.Series(vals).value_counts(normalize=True).values
        vals = pd.Series(vals).value_counts().to_dict()
        if labels is None:
            labels = [f"{value} ({count}) " for value, count in vals.items()]
    else:
        assert vals is None
        if labels is None:
            labels = [f"f{i}" for i in range(len(fractions))]

    if 'autopct' not in kwargs:
        kwargs['autopct'] = '%1.1f%%'

    ax.pie(fractions,
           labels=labels,
           **kwargs)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if title:
        plt.title(title, fontsize=20)

    if show:
        plt.show()

    return ax
