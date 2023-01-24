
__all__ = ["process_cbar", "make_cols_from_cmap", "process_axes",
           "kde", "make_clrs_from_cmap", "map_array_to_cmap"]

from typing import Union, Any, Optional, Tuple
from collections.abc import KeysView, ValuesView

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import RegularPolygon
from matplotlib.projections import register_projection
from mpl_toolkits.axes_grid1 import make_axes_locatable


BAR_CMAPS = ['Blues', 'BuGn', 'gist_earth_r',
             'GnBu', 'PuBu', 'PuBuGn', 'summer_r']


def process_axes(
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
        tight_layout:bool = False,
)-> plt.Axes:
    """
    processing of matplotlib Axes

    Parameters
    ----------
    ax : plt.Axes
        the :obj:`matplotlib.axes` axes which needs to be processed.
    label : str (default=None)
        will be used for legend
    legend_kws : dict, optional
        dictionary of keyword arguments to :obj:`matplotlib.axes.Axes.legend`
        These include loc, fontsize, bbox_to_anchor, markerscale
    logy : bool
        whether to convert y-axes to logrithmic scale or not
    logx : bool
        whether to convert x-axes to logrithmic scale or not
    xlabel : str
        label for x-axes
    xlabel_kws : dict
        keyword arguments for :obj:`matplotlib.axes.Axes.set_xlabel` ax.set_xlabel(xlabel, **xlabel_kws)
    xtick_kws :
        # for axes.tick_params such as which, labelsize, colors etc
    min_xticks : int
        maximum number of ticks on x-axes
    max_xticks : int
        minimum number of ticks on x-axes
    ylabel : str
        label for y-axes
    ylabel_kws : dict
        ylabel kwargs for :obj:`matplotlib.axes.Axes.set_ylabel`
    ytick_kws :
        for axes.tick_params()  such as which, labelsize, colors etc
    ylim :
        limit for y axes
    invert_yaxis :
        whether to invert y-axes or not. It true following command will be
        executed ax.set_ylim(ax.get_ylim()[::-1])
    title : str
        title for axes :obj:`matplotlib.axes.Axes.set_title`
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
    tight_layout : bool (default=False)
        whether to execulte plt.tight_layout() or not

    Returns
    -------
    plt.Axes
        the matplotlib Axes object :obj:`matplotlib.axes` which was passed to this function
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

    if top_spine is not None:
        ax.spines['top'].set_visible(top_spine)
    if bottom_spine is not None:
        ax.spines['bottom'].set_visible(bottom_spine)
    if right_spine is not None:
        ax.spines['right'].set_visible(right_spine)
    if left_spine is not None:
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

    if tight_layout:
        plt.tight_layout()

    return ax


def make_cols_from_cmap(
        cm: str,
        num_cols: int,
        low=0.0,
        high=1.0
)->np.ndarray:
    """make rgb colors from a color pallete"""
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
        try:
            array = np.array(array_like)
            assert len(array) == array.size
        except Exception:
            raise ValueError(f'cannot convert object {array_like.__class__.__name__}  to 1d ')
        return array

def has_multi_cols(data)->bool:
    """returns True if data contains multiple columns"""
    if hasattr(data, "ndim") and hasattr(data, "shape"):
        if data.ndim == 2 and data.shape[1]>1:
            return True
    return False


def kde(
        y,
        bw_method = "scott",
        bins:int = 1000,
        cut:Union[float, Tuple[float]] = 0.5,
)->Tuple[Union[np.ndarray, Tuple[np.ndarray, Optional[float]]], Any]:
    """
    Generate Kernel Density Estimate plot using Gaussian kernels.

    parameters
    ----------
    y :
    bw_method :
    bins :
    cut :
    """

    # don't want to make whole easy_mpl dependent upon scipy
    from scipy.stats import gaussian_kde

    if isinstance(cut, float):
        cut = (cut, cut)

    y = np.array(y)
    if 'int' not in y.dtype.name:  # 'object' types have problem in removing nans
        y = np.array(y, dtype=np.float32)

    assert len(y) == y.size
    y = y[~np.isnan(y)]
    gkde = gaussian_kde(y.reshape(-1,), bw_method=bw_method)

    sample_range = np.nanmax(y) - np.nanmin(y)
    ind = np.linspace(
        np.nanmin(y) - cut[0] * sample_range,
        np.nanmax(y) + cut[1] * sample_range,
        bins,
    )

    return ind, gkde.evaluate(ind)


def annotate_imshow(
        im,
        data:np.ndarray=None,
        textcolors:Union[tuple, np.ndarray]=("black", "white"),
        threshold=None,
        fmt = '%.2f',
        **text_kws
):
    """annotates imshow
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if data is None:
        data = im.get_array()

    use_threshold = True
    if isinstance(textcolors, np.ndarray) and textcolors.shape == data.shape:
        assert threshold is None, f"if textcolors is given as array then threshold should be None"
        use_threshold = False
    else:
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            s = fmt % float(data[i, j])
            if use_threshold:
                _ = im.axes.text(j, i, s,
                        color=textcolors[int(im.norm(data[i, j]) > threshold)],
                        **text_kws)
            else:
                _ = im.axes.text(j, i, s,
                        color=textcolors[i, j],
                        **text_kws)
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


def _rescale(y:np.ndarray, _min=0.0, _max=1.0)->np.ndarray:
    """rescales the y array between _min and _max"""
    y_std = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

    return y_std * (_max - _min) + _min


def version_info():
    import matplotlib
    from . import __version__
    info = dict()
    info['easy_mpl'] = __version__
    info['matplotlib'] = matplotlib.__version__
    info['numpy'] = np.__version__

    try:
        import pandas as pd
        info['pandas'] = pd.__version__
    except Exception:
        pass

    return info


def is_dataframe(obj)->bool:
    if all([hasattr(obj, attr) for attr in ["columns", "index", "values", "shape"]]):
        return True
    return False


def is_series(obj)->bool:
    if all([hasattr(obj, attr) for attr in ["name", "index", "values"]]):
        return True
    return False


def create_subplots(
        naxes:int,
        ax:plt.Axes = None,
        figsize:tuple = None,
        ncols:int=None,
        **fig_kws
)->Tuple:

    if ax is None:

        if naxes == 1:
            # if we need just one axes, just get the currently available one
            ax = plt.gca()
            fig = ax.get_figure()

        else:
            nrows, ncols = get_layout(naxes, ncols=ncols)
            plt.close('all')
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **fig_kws)
            switch_off_redundant_axes(naxes, nrows * ncols, ax)

    else:
        fig = ax.get_figure()
        if naxes == 1:
            return fig, ax

        nrows, ncols = get_layout(naxes)
        plt.close('all')
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **fig_kws)
        switch_off_redundant_axes(naxes, nrows*ncols, ax)

    return fig, ax


def switch_off_redundant_axes(naxes, nplots, axarr):

    shape = axarr.shape
    axarr = axarr.flatten()
    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    return axarr.reshape(shape)


def get_layout(naxes, ncols=None):

    if ncols and ncols==1:
        nrows = naxes
        return nrows, ncols

    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        nrows, ncols = layouts[naxes]
    except KeyError:
        k = 1
        while k ** 2 < naxes:
            k += 1

        if (k - 1) * k >= naxes:
            nrows, ncols = k, (k - 1)
        else:
            nrows, ncols =  k, k
    return nrows, ncols


class AddMarginalPlots(object):
    """
    Adds marginal plots for an axes.

    parameters
    -----------
    ax : plt.Axes
        :obj:`matplotlib.axes` on which to add the marginal plots
    pad :
    size :
    hist : bool
    hist_kws :
    ridge_line_kws :
    fill_kws :
    fix_limits : bool

    Examples
    ---------
    >>> from easy_mpl import plot
    >>> x = np.random.normal(size=100)
    >>> y = np.random.normal(size=100)
    >>> e = x-y
    >>> ax = plot(e, show=False)
    >>> AddMarginalPlots(ax, hist=True)(x, y)
    >>> plt.show()
    """
    def __init__(
            self,
            ax:plt.Axes,
            pad:float=0.25,
            size:float = 0.7,
            hist:bool = True,
            hist_kws:dict = None,
            ridge_line_kws:dict = None,
            fill_kws: dict = None,
            fix_limits:bool = True
    ):

        self.ax = ax

        if not isinstance(pad, (list, tuple)):
            pad = [pad, pad]
        self.pad = pad

        if not isinstance(size, (list, tuple)):
            size = [size, size]
        self.size = size

        self.hist = hist

        self.HIST_KWS = self.verify_kws(hist_kws)
        self.ridge_line_kws = self.verify_kws(ridge_line_kws)
        self.fill_kws = self.verify_kws(fill_kws)

        self.fix_limits = fix_limits



    def __call__(
            self,
            x,
            y,
            top_axes:plt.Axes=None,
            right_axes:plt.Axes=None
    )->tuple:
        """
        x : array like
        y : array like
        """
        self.divider = make_axes_locatable(self.ax)

        axHistx = self.add_ax_marg_x(x, hist_kws=self.HIST_KWS[0], ax=top_axes)
        axHisty = self.add_ax_marg_y(y_data=y, hist_kws=self.HIST_KWS[1], ax=right_axes)

        # make some labels invisible
        plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
                 visible=False)

        despine_axes(self.ax, keep=["left", "bottom"])

        return axHistx, axHisty

    def add_ax_marg_x(self,
                      x_data,
                      hist_kws:dict=None,
                      ax:plt.Axes = None,
                      ):

        line_kws = self._get_line_kws(self.ridge_line_kws[0])


        if self.fix_limits:
            xlim = np.array(self.ax.get_xlim()) * 1.05

        _hist_kws = {"linewidth":0.5, "edgecolor":"k"}
        if hist_kws is not None:
            _hist_kws.update(hist_kws)

        if ax is None:
            new_axes = self.divider.append_axes("top", self.size[0],
                                       pad=self.pad[0], sharex=self.ax)
        else:
            new_axes = ax

        despine_axes(new_axes, keep="bottom")
        new_axes.set_yticks([])

        if self.hist:
            # we draw histogram on old axes so that kde line
            # comes on top
            ax2 = new_axes.twinx()
            new_axes.hist(x_data, **_hist_kws)
            despine_axes(ax2)
            ax2.set_yticks([])
        else:
            ax2 = new_axes

        ind, data = kde(x_data, cut=0.2)
        ax2.plot(ind, data, **line_kws)

        if not self.hist:
            fill_kws = self._get_fill_kws(self.fill_kws[0], n=len(ind))

            ax2.fill_between(ind, data, **fill_kws)

        if self.fix_limits:
            self.ax.set_xlim(*xlim.tolist())

        return new_axes

    def add_ax_marg_y(self,
                      y_data,
                      hist_kws: dict = None,
                      ax:plt.Axes = None
                      ):

        line_kws = self._get_line_kws(self.ridge_line_kws[1])

        if self.fix_limits:
            ylim = self.ax.get_ylim()

        _hist_kws = {"linewidth":0.5,
                     "edgecolor":"k",
                     'orientation':'horizontal'}
        if hist_kws is not None:
            _hist_kws.update(hist_kws)

        if ax is None:
            new_axes = self.divider.append_axes(
                "right",
                self.size[1],
                pad=self.pad[1],
                sharey=self.ax)
        else:
            new_axes = ax

        despine_axes(new_axes, keep="left")
        new_axes.set_xticks([])

        if self.hist:
            ax2 = new_axes.twiny()
            new_axes.hist(y_data, **_hist_kws)
            despine_axes(ax2)
            ax2.set_xticks([])
        else:
            ax2 = new_axes

        ind, data = kde(y_data, cut=0.2)
        ax2.plot(data, ind, **line_kws)

        if not self.hist:
            fill_kws = self._get_fill_kws(self.fill_kws[1], n=len(ind))
            ax2.fill_betweenx(ind, data, **fill_kws)

        if self.fix_limits:
            self.ax.set_ylim(ylim)

        return new_axes

    @staticmethod
    def _get_line_kws(line_kws)->dict:
        _line_kws = {'color': 'k', 'lw': 1.0}
        if line_kws is not None:
            _line_kws.update(line_kws)
        return _line_kws

    @staticmethod
    def _get_fill_kws(fill_kws, n)->dict:
        _fill_kws = {"alpha": 0.5, 'color':'r',
                     'where': np.array([True for _ in range(n)])
                     }
        if fill_kws is not None:
            _fill_kws.update(fill_kws)
        return _fill_kws

    @staticmethod
    def verify_kws(kws=None):
        if kws is not None and not isinstance(kws, list):
            assert isinstance(kws, dict)
            kws = [kws, kws]

        elif kws is None:
            kws = [None, None]

        assert len(kws) == 2
        return kws


def despine_axes(axes, keep=None):

    if not isinstance(keep, list):
        keep = [keep]

    spines = ["top", "bottom", "right", "left"]
    for s in spines:
        if s not in keep:
            axes.spines[s].set_visible(False)
    return


def make_clrs_from_cmap(*args, **kwargs):
    return make_cols_from_cmap(*args, **kwargs)


def is_rgb(color)->bool:
    """returns True of ``color`` is rgb else returns False"""
    if isinstance(color, (list, np.ndarray, tuple)) and len(color) in [3,4] and isinstance(color[0], (int, float)):
        return True
    return False


def map_array_to_cmap(array, cmap:str, clip:bool = True):
    norm = Normalize(vmin=np.nanmin(array).item(),
                     vmax=np.nanmax(array).item(), clip=clip)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mapper.to_rgba(array)
    return colors, mapper


def process_cbar(
        ax,
        mappable,
        border:bool = True,
        width: Union[int, float] = None,
        pad: Union[int, float] = 0.2,
        orientation:str = "vertical",
        title:str = None,
        title_kws: dict = None
):
    """
    makes and processes colobar to an axisting axes

    Parameters
    ----------
    ax : plt.Axes
    mappable :
    border : bool
        wether to draw the border or not
    pad :
    orientation : str
    title : str
    title_kws : dict
        ``rotation``
        ``labelpad``
        ``fontsize``
        ``weight``

    Returns
    -------
    plt.colorbar
    """
    if orientation == "vertical":
        position = "right"
    else:
        position = "bottom"

    cb_params = {'pad': pad, 'orientation': orientation}
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=pad)
    cbar = plt.colorbar(mappable, cax=cax, **cb_params)

    if not border:
        # Turn spines off and create white grid.
        if isinstance(cax.spines, dict):
            for sp in cax.spines:
                cax.spines[sp].set_visible(False)
        else:
            cax.spines[:].set_visible(False)

    if title:
        if title_kws is None: title_kws = {}

        if orientation == "vertical":
            cbar.ax.set_ylabel(title, title_kws)
        else:
            cbar.ax.set_xlabel(title, **title_kws)
    return cbar
