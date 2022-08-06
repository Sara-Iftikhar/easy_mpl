
from collections.abc import KeysView, ValuesView

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection


BAR_CMAPS = ['Blues', 'BuGn', 'gist_earth_r',
             'GnBu', 'PuBu', 'PuBuGn', 'summer_r']

# colormaps for ridge plot
RIDGE_CMAPS = [
    "afmhot", "afmhot_r", "Blues", "bone",
    "BrBG", "BuGn", "coolwarm", "cubehelix",
    "gist_earth", "GnBu", "Greens", "magma",
    "ocean", "Pastel1", "pink", "PuBu", "PuBuGn",
    "RdBu", "Spectral",
]

regplot_combs = [
    ['cadetblue', 'slateblue', 'darkslateblue'],
    ['cadetblue', 'mediumblue', 'mediumblue'],
    ['cornflowerblue', 'dodgerblue', 'darkblue'],
    ['cornflowerblue', 'dodgerblue', 'steelblue'],
    ['cornflowerblue', 'mediumblue', 'dodgerblue'],
    ['cornflowerblue', 'steelblue', 'mediumblue'],
    ['darkslateblue', 'aliceblue', 'mediumblue'],
    ['darkslateblue', 'blue', 'royalblue'],
    ['darkslateblue', 'blueviolet', 'royalblue'],
    ['darkslateblue', 'darkblue', 'midnightblue'],
    ['darkslateblue', 'mediumblue', 'darkslateblue'],
    ['darkslateblue', 'midnightblue', 'mediumblue'],
    ['seagreen', 'darkslateblue', 'cadetblue'],
    ['cadetblue', 'darkblue', 'midnightblue'],
    ['cadetblue', 'deepskyblue', 'cadetblue']
]


def _regplot_paras(x, y, ci:int=None):
    """prepares parameters for regplot"""
    grid = np.linspace(np.min(x), np.max(x), 100)
    x = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(x, y))

    err_bands = None
    if ci:
        boots = bootdist(reg_func, args=[x, y], n_boot=1000).T

        yhat_boots = grid.dot(boots).T
        err_bands = _ci(yhat_boots, ci, axis=0)

    return grid, yhat, err_bands


def _regplot(x, y, ax, ci=None, line_color=None, fill_color=None):

    grid, yhat, err_bands = _regplot_paras(x, y, ci)

    ax.plot(grid[:, 1], yhat, color=line_color)

    if ci:
        ax.fill_between(grid[:, 1], *err_bands,
                        facecolor=fill_color,
                        alpha=.15)
    return ax


def _ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def reg_func(_x, _y):
    return np.linalg.pinv(_x).dot(_y)


def bootdist(f, args, n_boot=1000, **func_kwargs):

    n = len(args[0])
    integers = np.random.randint
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(f(*sample, **func_kwargs))

    return np.array(boot_dist)


def process_axis(
        ax: plt.Axes=None,
        label:str = None,
        legend_kws:dict = None,
        logy:bool = False,
        logx: bool= False,
        xlabel: str = None,
        xlabel_kws:dict=None,
        xtick_kws:dict = None,
        ylim:tuple = None,
        ylabel:str = None,
        ylabel_kws:dict = None,
        ytick_kws:dict = None,
        show_xaxis: bool = True,
        show_yaxis:bool = True,
        top_spine=None,
        bottom_spine=None,
        right_spine=None,
        left_spine=None,
        invert_yaxis: bool = False,
        max_xticks=None,
        min_xticks=None,
        title:str = None,
        title_kws:dict=None,
        grid=None,
        grid_kws:dict = None,
)-> plt.Axes:
    """
    processing of matplotlib Axes

    Parameters
    ----------
    ax : plt.Axes
        the axes which needs to be processed.
    label : str (default=None)
        will be used for legend
    legend_kws : dict, optional
        dictionary of keyword arguments to ax.legend(**legend_kws)
        These include loc, fontsize, bbox_to_anchor, markerscale
    logy : bool
    logx : bool
    xlabel : str
        label for x-axies
    xlabel_kws : dict
        keyword arguments for x-label ax.set_xlabel(xlabel, **xlabel_kws)
    xtick_kws :
        # for axes.tick_params such as which, labelsize, colors etc
    min_xticks :
    max_xticks :
    ylabel : str
    ylabel_kws : dict
        ylabel kwargs
    ytick_kws :
        for axes.tick_params()  such as which, labelsize, colors etc
    ylim :
        limit for y axes
    invert_yaxis :
        whether to invert y-axes or not. It true following command will be
        executed ax.set_ylim(ax.get_ylim()[::-1])
    title : str
    title_kws : dict
        title kwargs
    grid :
        will be fed to ax.grid(grid,...)
    grid_kws : dict
        dictionary of keyword arguments for ax.grid(grid, **grid_kws)
    left_spine :
    right_spine :
    top_spine :
    bottom_spine :
    show_xaxis : bool, optional (default=True)
        whether to show x-axes or not
    show_yaxis : bool, optional (default=True)
        whether to show y-axes or not

    Returns
    -------
    plt.Axes
        the matplotlib Axes object which was passed to this function
    """
    if ax is None:
        ax = plt.gca()

    if label:
        if label != "__nolabel__":
            legend_kws = legend_kws or {}
            ax.legend(**legend_kws)

    if ylabel:
        ylabel_kws = ylabel_kws or {}
        ax.set_ylabel(ylabel, **ylabel_kws)

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if invert_yaxis:
        ax.set_ylim(ax.get_ylim()[::-1])

    if ylim:
        ax.set_ylim(ylim)

    if xlabel:  # better not change these paras if user has not defined any x_label
        xtick_kws = xtick_kws or {}
        ax.tick_params(axis="x", **xtick_kws)

    if ylabel:
        ytick_kws = ytick_kws or {}
        ax.tick_params(axis="y", **ytick_kws)

    ax.get_xaxis().set_visible(show_xaxis)

    if xlabel:
        xlabel_kws = xlabel_kws or {}
        ax.set_xlabel(xlabel, **xlabel_kws)

    if top_spine:
        ax.spines['top'].set_visible(top_spine)
    if bottom_spine:
        ax.spines['bottom'].set_visible(bottom_spine)
    if right_spine:
        ax.spines['right'].set_visible(right_spine)
    if left_spine:
        ax.spines['left'].set_visible(left_spine)

    if max_xticks is not None:
        min_xticks = min_xticks or max_xticks-1
        assert isinstance(min_xticks, int)
        assert isinstance(max_xticks, int)
        loc = mdates.AutoDateLocator(minticks=min_xticks, maxticks=max_xticks)
        ax.xaxis.set_major_locator(loc)
        fmt = mdates.AutoDateFormatter(loc)
        ax.xaxis.set_major_formatter(fmt)

    if title:
        title_kws = title_kws or {}
        ax.set_title(title, **title_kws)

    if grid:
        grid_kws = grid_kws or {}
        ax.grid(grid, **grid_kws)

    if not show_yaxis:
        ax.get_yaxis().set_visible(False)

    return ax


def make_cols_from_cmap(cm: str, num_cols: int, low=0.0, high=1.0)->np.ndarray:

    cols = getattr(plt.cm, cm)(np.linspace(low, high, num_cols))
    return cols


def to_1d_array(array_like) -> np.ndarray:
    """returned array has shape (n,) """
    if array_like.__class__.__name__ in ['list', 'tuple', 'Series']:
        return np.array(array_like)

    elif array_like.__class__.__name__ == 'ndarray':
        if array_like.ndim == 1:
            return array_like
        else:
            assert array_like.size == len(array_like), f'cannot convert multidim ' \
                                                       f'array of shape {array_like.shape} to 1d'
            return array_like.reshape(-1, )

    elif array_like.__class__.__name__ == 'DataFrame' and array_like.ndim == 2:
        assert len(array_like) == array_like.size
        return array_like.values.reshape(-1,)

    elif isinstance(array_like, (float, int)):
        return np.array([array_like])

    elif isinstance(array_like, (KeysView, ValuesView)):
        return np.array(list(array_like))
    else:
        raise ValueError(f'cannot convert object {array_like.__class__.__name__}  to 1d ')


def has_multi_cols(data)->bool:
    """returns True if data contains multiple columns"""
    if isinstance(data, (pd.DataFrame, np.ndarray)):
        if data.ndim == 2 and data.shape[1]>1:
            return True
    return False


def kde(y):
    # don't want to make whole easy_mpl dependent upon scipy
    from scipy.stats import gaussian_kde

    """Generate Kernel Density Estimate plot using Gaussian kernels."""
    gkde = gaussian_kde(y, bw_method='scott')

    sample_range = np.nanmax(y) - np.nanmin(y)
    ind = np.linspace(
        np.nanmin(y) - 0.5 * sample_range,
        np.nanmax(y) + 0.5 * sample_range,
        1000,
    )

    return ind, gkde.evaluate(ind)


def annotate_imshow(
        im,
        data:np.ndarray=None,
        annotate_kws=None,
        textcolors=("black", "white"),
        threshold=None,
):
    """annotates imshow
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if data is None:
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2

    annotate_kws = annotate_kws or {"ha": "center", "va": "center"}
    if 'fmt' in annotate_kws:
        fmt = annotate_kws.pop('fmt')
    else:
        fmt = '%.2f'

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            s = fmt % float(data[i, j])
            _ = im.axes.text(j, i, s,
                        color=textcolors[int(im.norm(data[i, j]) > threshold)],
                        **annotate_kws)
    return


def register_projections(num_vars, frame="polygon", grids="polygon"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    grids: {"circle", "polygon"}

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                pass #return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
