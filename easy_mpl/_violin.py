
__all__ = ["violin_plot"]

import random
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt

from easy_mpl.utils import _rescale, kde


FILL_COLORS = [np.array([253,160,231])/255,
               np.array([102, 217, 191])/255,
               np.array([251, 173, 167])/255,
               np.array([220, 194, 102])/255,
               np.array([201, 185, 255])/255,
               ]


def violin_plot(
        data: Union[np.ndarray, List[np.ndarray]],
        X: Union[np.ndarray, List[np.ndarray]] = None,
        fill: bool = True,
        fill_colors=None,
        violin_kws: dict = None,
        show_datapoints: bool = True,
        datapoints_colors: Union[list, str] = None,
        scatter_kws: dict = None,
        show_boxplot: bool = False,
        box_kws: dict = None,
        label_violin: bool = False,
        index_method: str = "jitter",
        max_dots: int = 100,
        cut: Union[float, list] = 0.2,
        show: bool = True,
        ax: plt.Axes = None,
) -> plt.Axes:
    """

    parameters
    ----------
    data
    X :
        array or list of arrays. If list of arrays is given, the length of
        arrays can be unequal.
    fill : bool, optional (default=True)
        whether to fill the violin with color or not
    fill_colors :
    violin_kws : dict (default=None)
    show_datapoints : bool (default=True)
        whether to plot the datapoints or not
    datapoints_colors
    scatter_kws : dict (default=None)
        keyword arguments for axes.scatter. This will only be valid if
        ``show_datapoints`` is True.
    show_boxplot
    box_kws
        keyword arguments for axes.boxplot. This will only be valid if
        ``show_boxplot`` is True.
    label_violin
    index_method : str (default="jitter")
        Only valid if `X` is not given. The method to generate indices for x-axis.
        See `<https://stackoverflow.com/a/33965400/5982232> this_` for context
    max_dots : int (default=100)
        maximum number of dots to show
    cut : float/list (default=0.2)
        This variables determines the length of violin. If given as a list, it should
        match the number of arrays in X.
    show : bool (default=True)
        whether to show the plot or not
    ax : plt.Axes (default=None)


    Returns
    -------
    plt.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl._violin import violin_plot
    >>> data = np.random.gamma(20, 10, 100)
    >>> violin_plot(data)
    >>> violin_plot(data, show_datapoints=False)
    >>> violin_plot(data, show_datapoints=False, show_boxplot=True)
    """

    names = None
    if isinstance(data, np.ndarray):
        Y = to_1d_arrays(data)

    elif hasattr(data, "values") and hasattr(data, "columns"):   # data is pd.DataFrame
        Y = to_1d_arrays(data.values)
        names = data.columns.tolist()
    else:
        assert isinstance(data, list), f"Unrecognized data of type {data.__class__.__name__}"
        Y = data

    if X is None:
        if index_method == "jitter":
            X = jittered_ind(Y)
            offsets = [0] * len(Y)
        else:
            maxs = get_maxs(Y, cut=cut)
            X, offsets = [], []
            for idx, y in enumerate(Y):
               x = beeswarm_ind(y, nbins=100, lim=maxs[idx]*0.8)
               X.append(x)
               offsets.append(idx)
    else:
        offsets = [0] * len(Y)


    if not isinstance(X, list):
        X = [X]

    assert isinstance(X, list)
    assert len(X) == len(Y)

    # Horizontal positions for the violins.
    # They are arbitrary numbers. They could have been [-1, 0, 1] for example.
    POSITIONS = range(len(X))

    if ax is None:
        ax = plt.gca()

    # Add violins ----------------------------------------------------
    # bw_method="silverman" means the bandwidth of the kernel density
    # estimator is computed via Silverman's rule of thumb.
    # More on this in the bonus track ;)

    _violin_kws = {
        "widths":0.45,
        "bw_method": "silverman",
        "showmeans": False,
        "showmedians": False,
        "showextrema": False
    }

    if violin_kws is None:
        violin_kws = dict()

    _violin_kws.update(violin_kws)

    vpstats = get_vpstats(Y,
                          bw_method=_violin_kws.get('bw_method', None),
                          cut=cut)

    if "bw_method" in _violin_kws:
        _violin_kws.pop("bw_method")

    # The output is stored in 'violins', used to customize their appearence
    violins = ax.violin(vpstats, positions=POSITIONS, **_violin_kws)

    if isinstance(fill_colors, str):
        fill_colors = [fill_colors for _ in range(len(X))]

    if fill_colors is None:
        fill_colors = random.choices(FILL_COLORS, k=len(X))

    # Customize violins (remove fill, customize line, etc.)
    for idx, pc in enumerate(violins["bodies"]):
        if fill:
            pc.set_facecolor(fill_colors[idx])
        else:
            pc.set_facecolor("none")
        pc.set_edgecolor("#282724")
        pc.set_alpha(1)

    # Add boxplots
    if show_boxplot:

        boxprops = dict(
            linewidth=2,
            color="#747473"  # GREY_DARK
        )

        _box_kws = {
            'medianprops': dict(
                linewidth=4,
                color="#747473",  # GREY_DARK
                solid_capstyle="butt"
            ),
            'boxprops': boxprops,
            'whiskerprops': boxprops,
            'showfliers': False,  # Do not show the outliers beyond the caps.
            'showcaps': False,  # Do not show the caps
        }

        if box_kws is None:
            box_kws = dict()

        _box_kws.update(box_kws)
        ax.boxplot(Y, positions=POSITIONS, **_box_kws)

    # Add  dots
    if show_datapoints:
        _scatter_kws = {
            'alpha': 0.4,
            's': 10
        }

        if scatter_kws is None:
            scatter_kws = dict()

        _scatter_kws.update(scatter_kws)

        if isinstance(datapoints_colors, str):
            datapoints_colors = [datapoints_colors for _ in range(len(X))]

        if datapoints_colors is None:
            datapoints_colors = ["black" for _ in range(len(X))]

        for offset, x, y, color in zip(offsets, X, Y, datapoints_colors):

            x, y = sample(x+offset, y, n=max_dots)

            ax.scatter(x, y, color=color, **_scatter_kws)

    if label_violin:
        # Add mean value labels ------------------------------------------
        means = [y.mean() for y in Y]
        for i, mean in enumerate(means):
            # Add dot representing the mean
            ax.scatter(i, mean, s=250, color="#850e00",  # dark red
                       zorder=3)

            # Add line conecting mean value and its label
            ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)

            # Add mean value label.
            ax.text(
                i + 0.25,
                mean,
                r"$\hat{\mu}_{\rm{mean}} = $" + str(round(mean, 2)),
                fontsize=13,
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round",
                    pad=0.15
                ),
                zorder=10  # to make sure the line is on top
            )

    if names is not None:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)

    if show:
        plt.show()

    return ax


def jittered_ind(y, jitter=0.04):
    import scipy.stats as st

    x_data = [np.array([i] * len(d)) for i, d in enumerate(y)]
    return [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]


def beeswarm_ind(y, nbins=None, lim=1.0):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    ind = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            ind[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            ind[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return _rescale(ind, -lim, lim)


def get_vpstats(dataset, bins=100, cut=0.2, bw_method=None):
    if isinstance(cut, float):
        cut = [cut for _ in range(len(dataset))]

    assert isinstance(cut, list)
    assert len(cut)==len(dataset)

    vpstats = []

    for idx, y in enumerate(dataset):
        stats = dict()
        coords, vals = kde(y, bw_method=bw_method, bins=bins, cut=cut[idx])
        stats['coords'] = coords
        stats['vals'] = vals
        stats['mean'] = np.mean(y)
        stats['median'] = np.median(y)
        stats['min'] = np.min(y)
        stats['max'] = np.max(y)
        vpstats.append(stats)

    return vpstats


def get_maxs(dataset, points=100, quantiles=None, bw_method=None, cut=0.5):

    vpstats = get_vpstats(dataset,
                          bins=points, #quantiles=quantiles,
                          bw_method=bw_method,
                          cut=cut)

    N = len(vpstats)
    widths = [0.45] * N

    mins, maxs = [], []
    for stats, width in zip(vpstats, widths):

        vals = np.array(stats['vals'])
        vals = 0.5 * width * vals / vals.max()
        #mins.append(np.min(vals))
        maxs.append(np.max(vals))

    return maxs


def sample(x, y, n=100):
    if len(x)<=n:
        return x, y
    idx = np.random.choice(np.arange(len(x)), n, replace=False)

    return x[idx], y[idx]


def to_1d_arrays(data)->List[np.ndarray,]:
    if len(data) == data.size:
        Y = [data.reshape(-1,)]
    else:
        assert data.ndim == 2
        Y = [data[:, i] for i in range(data.shape[1])]
    return Y