
__all__ = ["violin_plot"]

import random
from typing import Union, List

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from easy_mpl.utils import _rescale, kde, is_dataframe


FILL_COLORS = [np.array([253,160,231])/255,
               np.array([102, 217, 191])/255,
               np.array([251, 173, 167])/255,
               np.array([220, 194, 102])/255,
               np.array([201, 185, 255])/255,
               np.array([205, 255, 205])/255,
               np.array([129, 207, 128])/255,
               np.array([103, 199, 209])/255,
               np.array([176, 204, 255])/255,
               ]


# todo, add orientation

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
        max_dots: Union[int, List[int]] = 100,
        cut: Union[float, tuple, List[float], List[tuple]] = 0.2,
        labels: Union[str, List[str]] = None,
        ax: plt.Axes = None,
        show: bool = True,
) -> plt.Axes:
    """
    makes violin plot/plots of the arrays in data

    parameters
    ----------
    data :
        It can array like or list of arrays. The length of each array need
        not be equal. If multiple arrays are given, then violin is
        drawn for each array
    X :
        indices for x-axes for each of the array in data. It can be
        array or list of arrays. If list of arrays is given, the length of
        arrays can be unequal.
    fill : bool, optional (default=True)
        whether to fill the violin with color or not
    fill_colors :
        colors to fill the violins
    violin_kws : dict (default=None)
        any keyword arugment for axes.violin_ in the form of dictionary
    show_datapoints : bool (default=True)
        whether to plot the datapoints or not
    datapoints_colors
        color for the datapoints/markers
    scatter_kws : dict (default=None)
        keyword arguments for axes.scatter_. This will only be valid if
        ``show_datapoints`` is True.
    show_boxplot : bool (default=False)
        whether to show the boxplot inside the voilin or not?
    box_kws : dict (default=None)
        keyword arguments for :obj:`matplotlib.axes.Axes.boxplot`. This will only be valid if
        ``show_boxplot`` is True.
    label_violin : bool (default=False)
        whether to label mean value of each violin or not
    index_method : str (default="jitter")
        Only valid if `X` is not given. The method to generate indices for x-axis.
        See `this <https://stackoverflow.com/a/33965400/5982232>_` for context
    max_dots : int/list (default=100)
        maximum number of dots to show. It can also be a list of integers, which
        would define the number of dots for each array in X.
    cut : float/list (default=0.2)
        This variables determines the length of violin. If given as a list or
        list of tuples, it should match the number of arrays in X.
    labels : list/str (default=None
        names for xticks
    ax : plt.Axes (default=None)
        matplotlib Axes object :obj:`matplotlib.axes` on which to draw the plot. If not given, then
        the currently available axes from plt.gca will be used.
    show : bool (default=True)
        whether to show the plot or not

    Returns
    -------
    plt.Axes
        the matplotlib Axes object on which violin/violins are drawn

    Examples
    --------
    >>> import numpy as np
    >>> from easy_mpl import violin_plot
    >>> data = np.random.gamma(20, 10, 100)
    >>> violin_plot(data)
    >>> violin_plot(data, show_datapoints=False)
    >>> violin_plot(data, show_datapoints=False, show_boxplot=True)

    See :ref:`sphx_glr_auto_examples_violin.py` for more examples

    .. _axes.scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html

    .. _axes.violin:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violin.html
    """

    if isinstance(data, np.ndarray):
        Y = to_1d_arrays(data)

    elif is_dataframe(data):   # data is pd.DataFrame
        Y = to_1d_arrays(data.values)
        if labels is None:
            labels = data.columns.tolist()
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
    if matplotlib.__version__ <= "3.4.0":
        violins = violin_mpl_340(ax, vpstats, positions=POSITIONS, **_violin_kws)
    else:
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

        if isinstance(max_dots, int):
            max_dots = [max_dots for _ in range(len(X))]

        for offset, x, y, color, n_dots in zip(offsets, X, Y, datapoints_colors, max_dots):

            x, y = sample(x+offset, y, n=n_dots)

            ax.scatter(x, y, color=color, **_scatter_kws)

    if label_violin:
        # Add mean value labels ------------------------------------------
        means = [y.mean() for y in Y]
        _mean_kws = {
            's': 12,
            'color': '#850e00', # dark red
            'zorder': 3
        }
        for i, mean in enumerate(means):
            # Add dot representing the mean
            ax.scatter(i, mean, **_mean_kws)

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

    if labels is not None:
        kws = dict()
        if len(labels)>7:
            kws['rotation'] = 90
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, **kws)

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

    if isinstance(cut, (float, tuple)):
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

# ******************************************
# Following code is taken from matplotlib
# because for versions prior to 3.5 (3.4 or prior versions)
# we need to set where argument to fill_between otherwise
# fill_between returns error.
# matplotlib comes with following licence
"""
License agreement for matplotlib versions 1.3.0 and later
=========================================================

1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and
otherwise using matplotlib software in source or binary form and its
associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that MDT's
License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012- Matplotlib Development Team; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib .

4. MDT is making matplotlib available to Licensee on an "AS
IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between MDT and
Licensee.  This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib ,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.
"""


def violin_mpl_340(self, vpstats, positions=None, vert=True, widths=0.5,
           showmeans=False, showextrema=True, showmedians=False):
    """
    Drawing function for violin plots.

    Draw a violin plot for each column of *vpstats*. Each filled area
    extends to represent the entire data range, with optional lines at the
    mean, the median, the minimum, the maximum, and the quantiles values.

    Parameters
    ----------
    vpstats : list of dicts
      A list of dictionaries containing stats for each violin plot.
      Required keys are:

      - ``coords``: A list of scalars containing the coordinates that
        the violin's kernel density estimate were evaluated at.

      - ``vals``: A list of scalars containing the values of the
        kernel density estimate at each of the coordinates given
        in *coords*.

      - ``mean``: The mean value for this violin's dataset.

      - ``median``: The median value for this violin's dataset.

      - ``min``: The minimum value for this violin's dataset.

      - ``max``: The maximum value for this violin's dataset.

      Optional keys are:

      - ``quantiles``: A list of scalars containing the quantile values
        for this violin's dataset.

    positions : array-like, default: [1, 2, ..., n]
      The positions of the violins. The ticks and limits are
      automatically set to match the positions.

    vert : bool, default: True.
      If true, plots the violins vertically.
      Otherwise, plots the violins horizontally.

    widths : array-like, default: 0.5
      Either a scalar or a vector that sets the maximal width of
      each violin. The default is 0.5, which uses about half of the
      available horizontal space.

    showmeans : bool, default: False
      If true, will toggle rendering of the means.

    showextrema : bool, default: True
      If true, will toggle rendering of the extrema.

    showmedians : bool, default: False
      If true, will toggle rendering of the medians.

    Returns
    -------
    dict
      A dictionary mapping each component of the violinplot to a
      list of the corresponding collection instances created. The
      dictionary has the following keys:

      - ``bodies``: A list of the `~.collections.PolyCollection`
        instances containing the filled area of each violin.

      - ``cmeans``: A `~.collections.LineCollection` instance that marks
        the mean values of each of the violin's distribution.

      - ``cmins``: A `~.collections.LineCollection` instance that marks
        the bottom of each violin's distribution.

      - ``cmaxes``: A `~.collections.LineCollection` instance that marks
        the top of each violin's distribution.

      - ``cbars``: A `~.collections.LineCollection` instance that marks
        the centers of each violin's distribution.

      - ``cmedians``: A `~.collections.LineCollection` instance that
        marks the median values of each of the violin's distribution.

      - ``cquantiles``: A `~.collections.LineCollection` instance created
        to identify the quantiles values of each of the violin's
        distribution.

    """

    # Statistical quantities to be plotted on the violins
    means = []
    mins = []
    maxes = []
    medians = []
    quantiles = np.asarray([])

    # Collections to be returned
    artists = {}

    N = len(vpstats)
    datashape_message = ("List of violinplot statistics and `{0}` "
                         "values must have the same length")

    # Validate positions
    if positions is None:
        positions = range(1, N + 1)
    elif len(positions) != N:
        raise ValueError(datashape_message.format("positions"))

    # Validate widths
    if np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format("widths"))

    # Calculate ranges for statistics lines
    pmins = -0.25 * np.array(widths) + positions
    pmaxes = 0.25 * np.array(widths) + positions

    # Check whether we are rendering vertically or horizontally
    if vert:
        fill = self.fill_betweenx
        perp_lines = self.hlines
        par_lines = self.vlines
    else:
        fill = self.fill_between
        perp_lines = self.vlines
        par_lines = self.hlines

    if rcParams['_internal.classic_mode']:
        fillcolor = 'y'
        edgecolor = 'r'
    else:
        fillcolor = edgecolor = self._get_lines.get_next_color()

    # Render violins
    bodies = []
    for stats, pos, width in zip(vpstats, positions, widths):
        # The 0.5 factor reflects the fact that we plot from v-p to
        # v+p
        vals = np.array(stats['vals'])
        vals = 0.5 * width * vals / vals.max()
        bodies += [fill(stats['coords'],
                        -vals + pos,
                        vals + pos,
                        facecolor=fillcolor,
                        where = np.array([True for _ in range(len(stats['coords']))]),
                        alpha=0.3)]
        means.append(stats['mean'])
        mins.append(stats['min'])
        maxes.append(stats['max'])
        medians.append(stats['median'])
        q = stats.get('quantiles')
        if q is not None:
            # If exist key quantiles, assume it's a list of floats
            quantiles = np.concatenate((quantiles, q))
    artists['bodies'] = bodies

    # Render means
    if showmeans:
        artists['cmeans'] = perp_lines(means, pmins, pmaxes,
                                       colors=edgecolor)

    # Render extrema
    if showextrema:
        artists['cmaxes'] = perp_lines(maxes, pmins, pmaxes,
                                       colors=edgecolor)
        artists['cmins'] = perp_lines(mins, pmins, pmaxes,
                                      colors=edgecolor)
        artists['cbars'] = par_lines(positions, mins, maxes,
                                     colors=edgecolor)

    # Render medians
    if showmedians:
        artists['cmedians'] = perp_lines(medians,
                                         pmins,
                                         pmaxes,
                                         colors=edgecolor)

    # Render quantile values
    if quantiles.size > 0:
        # Recalculate ranges for statistics lines for quantiles.
        # ppmins are the left end of quantiles lines
        ppmins = np.asarray([])
        # pmaxes are the right end of quantiles lines
        ppmaxs = np.asarray([])
        for stats, cmin, cmax in zip(vpstats, pmins, pmaxes):
            q = stats.get('quantiles')
            if q is not None:
                ppmins = np.concatenate((ppmins, [cmin] * np.size(q)))
                ppmaxs = np.concatenate((ppmaxs, [cmax] * np.size(q)))
        # Start rendering
        artists['cquantiles'] = perp_lines(quantiles, ppmins, ppmaxs,
                                           colors=edgecolor)

    return artists