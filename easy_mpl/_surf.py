
__all__ = ["surf"]

from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

def surf(
        x,
        y,
        z: Union[np.ndarray, Tuple[np.ndarray]] = None,
        c: Union[np.ndarray, Tuple[np.ndarray]] = None,
        facecolor = None,
        edgecolor = None,
        alpha = None,
        ax=None,
        show=True,
)->plt.Axes:
    """
    Plots surface. The purpose is to ease the drawing of surfaces
    using matplotlib and replicate matlab's surf function.

    parameters
    ----------
    x :
    y :
    z :
    c : Optional (default=None)
        array defining color of surface
    facecolor :
    edgecolor :
    alpha :
    ax : plt.Axes, optional (default=None)
        the matplotlib Axes object on which to plot the surface.
        If nto given, current available axes is taken using plt.gca()
    show : bool, optional (default=None)
        whether to show the plot or not.

    Returns
    --------
    plt.Axes
        matplotlib axes object on which surface is drawn

    Examples
    --------
    >>> from easy_mpl import surf
    >>> X = np.random.random(100)
    >>> Y = np.random.random(100)
    >> surf(X, Y)
    >>> Z = np.random.random(100)
    >>> surf(X, Y, Z)
    >>> C = np.random.random(100)
    >>> surf(X, Y, Z, C)
    """
    if not ax:
        ax = plt.gca()

    if show:
        plt.show()

    raise NotImplementedError
