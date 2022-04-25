
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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
        label=None,  # legend
        logy=False,
        logx=False,
        legend_kws:dict = None, # kwargs for axes.legend such as loc, fontsize, bbox_to_anchor, markerscale
        xlabel=None,
        xlabel_kws:dict=None,
        xtick_kws:dict = None, # for axes.tick_params such as which, labelsize, colors etc
        ylim:tuple=None,  # limit for y axes
        ylabel=None,
        ylabel_kws:dict=None,  # ylabel kwargs
        ytick_kws:dict = None, # for axes.tick_params(  such as which, labelsize, colors etc
        show_xaxis=True,
        top_spine=None,
        bottom_spine=None,
        right_spine=None,
        left_spine=None,
        invert_yaxis=False,
        max_xticks=None,
        min_xticks=None,
        title=None,
        title_kws:dict=None,  # title kwargs
        grid=None,
        grid_kws:dict = None,  # keyword arguments for axes.grid
)-> plt.Axes:
    """
    processing of matplotlib Axes
    Parameters
    ----------
    ax : plt.Axes
    label
    logy : bool
    logx : bool
    legend_kws :
    xlabel :
    xlabel_kws :
    xtick_kws :
    min_xticks :
    max_xticks :
    ylabel :
    ylabel_kws :
    ytick_kws :
    ylim :
    invert_yaxis :
    title :
    title_kws :
    grid :
    grid_kws :
    left_spine :
    right_spine :
    top_spine :
    bottom_spine :
    show_xaxis :

    Returns
    -------
        plt.Axes
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

    return ax


def get_cmap(cm: str, num_cols: int, low=0.0, high=1.0):

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
    elif isinstance(array_like, float) or isinstance(array_like, int):
        return np.array([array_like])
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
