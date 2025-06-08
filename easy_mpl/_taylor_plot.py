__all__ = ['taylor_plot']

from typing import Union
import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.grid_finder as GF
import mpl_toolkits.axisartist.floating_axes as FA

from easy_mpl.utils import is_dataframe, is_series

COLORS = {
    0: np.array([0.89411765, 0.10196078, 0.10980392, 1.]),
    1: np.array([0.21568627, 0.49411765, 0.72156863, 1.]),
    2: np.array([0.30196078, 0.68627451, 0.29019608, 1.]),
    3: np.array([0.59607843, 0.30588235, 0.63921569, 1.]),
    4: np.array([0.95896577, 0.58394066, 0.04189788, 1.]),
    5: np.array([0.96862745, 0.50588235, 0.74901961, 1.]),
    6: np.array([0.17877267, 0.78893675, 0.92613355, 1.]),
    7: np.array([0.38079258, 0.17830983, 0.78165943, 1.]),
    8: np.array([0.13778617, 0.06228198, 0.33547859, 1.]),
    9: np.array([0.96707953, 0.46268314, 0.45772886, 1.]),
    10: np.array([0.17221373, 0.53023578, 0.96788307, 1.]),
    11: np.array([0.92875036, 0.09364162, 0.33348078, 1.]),
    12: np.array([0.93950089, 0.64582256, 0.16928645, 1.]),
    13: np.array([0.06802773, 0.46382623, 0.49007703, 1.]),
    14: np.array([0.13684922, 0.98802401, 0.34518303, 1.]),
    15: np.array(np.random.random(3).tolist() + [1.]),
    16: np.array([0.54829269, 0.15069842, 0.06147751, 1.]),
    17: np.array([0.61, 0.59, 0.51, 1.]),
    18: np.array([0.54829269, 0.15069842, 0.06147751, 1.]),
    19: np.array([0.06802773, 0.46382623, 0.49007703, 1.]),
    20: np.array([0.80, 0.50, 0.59, 1.]),
    21: np.array(np.random.random(3).tolist() + [1.]),
    22: np.array(np.random.random(3).tolist() + [1.]),
    23: np.array(np.random.random(3).tolist() + [1.]),
    24: np.array(np.random.random(3).tolist() + [1.]),
    25: np.array(np.random.random(3).tolist() + [1.]),
    26: np.array(np.random.random(3).tolist() + [1.]),
    27: np.array(np.random.random(3).tolist() + [1.]),
    28: np.array(np.random.random(3).tolist() + [1.]),
    29: np.array(np.random.random(3).tolist() + [1.]),
    30: np.array(np.random.random(3).tolist() + [1.]),
    31: np.array(np.random.random(3).tolist() + [1.]),
    32: np.array(np.random.random(3).tolist() + [1.]),
    33: np.array(np.random.random(3).tolist() + [1.]),
    34: np.array(np.random.random(3).tolist() + [1.]),
    35: np.array(np.random.random(3).tolist() + [1.]),
    36: np.array(np.random.random(3).tolist() + [1.]),
    37: np.array(np.random.random(3).tolist() + [1.]),
    38: np.array(np.random.random(3).tolist() + [1.]),
    39: np.array(np.random.random(3).tolist() + [1.]),
    40: np.array(np.random.random(3).tolist() + [1.]),
    41: np.array(np.random.random(3).tolist() + [1.]),
    42: np.array(np.random.random(3).tolist() + [1.]),
    43: np.array(np.random.random(3).tolist() + [1.]),
    44: np.array(np.random.random(3).tolist() + [1.]),
    45: np.array(np.random.random(3).tolist() + [1.]),
    46: np.array(np.random.random(3).tolist() + [1.]),
    47: np.array(np.random.random(3).tolist() + [1.]),
    48: np.array(np.random.random(3).tolist() + [1.]),
    49: np.array(np.random.random(3).tolist() + [1.]),
    50: np.array(np.random.random(3).tolist() + [1.]),
    51: np.array(np.random.random(3).tolist() + [1.]),
    52: np.array(np.random.random(3).tolist() + [1.]),
}

RECTS = {1: (111,),
         2: (211, 212),
         3: (311, 312, 313),
         4: (221, 222, 223, 224)}


def corr_coeff(t, p) -> float:
    """correlation coefficient"""
    return np.corrcoef(t, p)[0, 1]


def pbias(t, p) -> float:
    """percent bias"""
    return np.mean(t - p) / np.mean(p)


class TaylorDiagram(object):
    """
    Taylor diagram.
    Modified after https://gist.github.com/ycopin/3342888
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self,
                 true,
                 fig=None,
                 rect=111,
                 label='_',
                 srange=(0, 1.5),
                 extend=False,
                 axis_fontdict: dict = None,
                 name: str = '',
                 corr_alias:str = "Correlation",
                 std_alias:str = "Standard Deviation"
                 ):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
            true : true array or a dictionary containing standard deviation of the true
            fig : input Figure or None
            rect : subplot definition
            label : reference label
            srange : stddev axis extension, in units of *refstd*
            extend : extend diagram to negative correlations
            axis_fontdict : dictionary must consist of at least three dictionaries 'left', 'right', 'top'
        """

        # Reference standard deviation
        if isinstance(true, dict):
            self.refstd = true['std']
        else:
            self.refstd = np.std(true)

        if hasattr(self.refstd, 'item'):  # if true is series/dataframe
            self.refstd = self.refstd.item()

        assert not np.isnan(self.refstd), f"""
        stdandard deviation of {name} Observations is nan."""

        tr = PolarAxes.PolarTransform(apply_theta_transforms=False)

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text(corr_alias)
        ax.axis['top'].label.set_fontsize(axis_fontdict['top'].get('fontsize', 18))
        ax.axis['top'].label.set_color(axis_fontdict['top'].get('color', 'k'))
        ax.axis['top'].major_ticklabels.set_fontsize(axis_fontdict['top'].get('ticklabel_fs', 10))

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        if extend:
            ax.text(0.5, -0.12, std_alias, transform=ax.transAxes,
                    rotation=0, va='center', ha='center', fontdict={'size': axis_fontdict['top'].get('fontsize', 18)})
        else:
            ax.axis["left"].label.set_text(std_alias)
        ax.axis['left'].label.set_fontsize(axis_fontdict['bottom'].get('fontsize', 18))
        ax.axis['left'].label.set_color(axis_fontdict['bottom'].get('color', 'k'))
        ax.axis['left'].major_ticklabels.set_fontsize(axis_fontdict['bottom'].get('ticklabel_fs', 10))

        ax.axis["right"].set_axis_direction("top")  # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        if not extend:
            ax.text(-0.08, 0.5, std_alias, transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontdict={'size': axis_fontdict['top'].get('fontsize', 18)})
        ax.axis['right'].label.set_fontsize(axis_fontdict['left'].get('fontsize', 18))
        ax.axis['right'].label.set_color(axis_fontdict['left'].get('color', 'k'))
        ax.axis['right'].major_ticklabels.set_fontsize(axis_fontdict['left'].get('ticklabel_fs', 10))
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)  # Unused

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


def taylor_plot(
        observations: Union[dict, np.ndarray],
        simulations: Union[dict, np.ndarray],
        axis_locs: dict = None,
        cont_kws: dict = None,
        grid_kws: dict = None,
        leg_kws: dict = None,
        axis_fontdict: dict = None,
        axis_kws: dict = None,
        marker_kws: dict = None,
        **kwargs
) -> plt.Figure:
    """
    Helper function to plot `Taylor`_'s plot. A taylor plot is useful to compare
    results of different simulations.

    Arguments:
        observations : dict, array
            If it is an array then it means we have only one scenario. If it is
            a dictionary of length > 1, then its keys are scenarios and values
            represent true/observations at that scenarios. The values can also
            be a dictionary containing `std`, which stands for standard deviation.
        simulations : dict
            A dictionary of length > 1 whose keys are scenarios and whose values
            are also dictionary. Each sub-dictionary i.e. dictionary of scenario
            consist of models/simulations.
        axis_locs : dict, optional
            dictionary defining axis orientation of figure. For example with two
            scenarios named 'scenario1' and 'scenario2', if we want to plot two
            plots in one column, then this argument will be

            >>>          {'scenario1': 211,
            >>>           'scenario2': 212}.

            Default is None.
        cont_kws : dict, optional
            keyword arguments related to `contours`_. Following args can be used:

                - ``levels level`` of contours
                - ``colors color`` of contours
                - ``label_fs`` fontsize of labels
                - ``label_fmt`` format of labels
                - ``linewidths`` float or sequence of floats
                - ``linestyles`` {None, 'solid', 'dashed', 'dashdot', 'dotted'}

        grid_kws : dict, optional
            keyword arguments related to grid. Following args can be used.
            Following keyword arguments are allowed:

                - title_fontsize: int, fontsize of the axis title
                - which {'major', 'minor', 'both'}
                - axis {'both', 'x', 'y'},
            any kwargs from `axes.grid`_ are allowed.

        leg_kws : dict, optional
            keyword arguments related to legends:

                - position defaults to `center`
                - fontsize int or {'xx-small', 'x-small', 'small', 'medium', 'large',
                                   'x-large', 'xx-large'}
                - numpoints int, default: rcParams["legend.numpoints"] (default: 1)
                - markerscale float, default: rcParams["legend.markerscale"] (default: 1.0)
            see `axes.legend`_ for more details.

            >>> example leg_kws = {'loc': 'upper right', 'numpoints': 1, 'fontsize': 15,
            ...                    'markerscale': 1}

        axis_fontdict : dict, optional
            dictionary defining propertiies of left, bottom and top axis labels

            >>> axis_fontdict = {'left': {'fontsize': 20, 'color': 'k', 'ticklabel_fs': 14},
            ...                 'bottom': {'fontsize': 20, 'color': 'g', 'ticklabel_fs': 14},
            ...                 'top': {'fontsize': 20, 'color': 'k', 'ticklabel_fs': 14}}

            The user can define properties of either one or all axis.

        axis_kws : dict, optional
            dictionary containing general parameters related to axis such as title.

        marker_kws : dict, optional
            dictionary containing marker properties. All of these keywords are passed to
            `matplotlib.axes.Axes.plot`.

        **kwargs :
            Following keyword arguments are optional:

            - add_ith_interval : bool

            - plot_bias : bool,
                if True, the size of the markers will be used to
                represent bias. The markers will be triangles with their sides up/down
                depending upon value of bias.

            - ref_color : str,
                color of refrence dot

            - sim_marker :
                marker to use for simulations. It can be any valid
                marker for matplotlib axis/plot. If None, then counting is used.
                If string, then same marker is used for all simulations. If dict,
                keys of dict should match with names of models in `simulations` dictionary.

            - true_label : label to use for `trues`. Default is 'Reference'.

            - intervals : list, if add_ith_interval is True, then this argument is used. It
                must be list of lists or list of tuples, where the inner tuple/list must
                consist of two values one each for x and y.

            - colors : 2d numpy array, defining colors. The first dimension
                should be equal to number of models.

            - extend : bool, default False, if True, will plot negative correlation

            - corr_alias : str,
                alias for correlation (text for curved axis label), default is 'Correlation'

            - std_alias : str,
                alias for standard deviation (text for horizontal and vertical
                axis label), default is 'Standard Deviation'

            - figsize : tuple defining figsize, default is (11,8).

            - show : bool whether to show the plot or not
    Returns:
        matplotlib figure

    Example:
        >>> import numpy as np
        >>> from easy_mpl import taylor_plot
        >>> np.random.seed(313)
        >>> observations =  np.random.normal(20, 40, 10)
        >>> simulations =  {"LSTM": np.random.normal(20, 40, 10),
        ...             "CNN": np.random.normal(20, 40, 10),
        ...             "TCN": np.random.normal(20, 40, 10),
        ...             "CNN-LSTM": np.random.normal(20, 40, 10)}
        >>> taylor_plot(observations=observations,
        ...             simulations=simulations,
        ...             title="Taylor Plot")
        ...
        ... # multiple taylor plots in one figure
        >>> np.random.seed(313)
        >>> observations = {
        ...     'site1': np.random.normal(20, 40, 10),
        ...     'site2': np.random.normal(20, 40, 10),
        ...     'site3': np.random.normal(20, 40, 10),
        ...     'site4': np.random.normal(20, 40, 10),
        ... }
        ...
        >>> simulations = {
        ...     "site1": {"LSTM": np.random.normal(20, 40, 10),
        ...                 "CNN": np.random.normal(20, 40, 10),
        ...                 "TCN": np.random.normal(20, 40, 10),
        ...                 "CNN-LSTM": np.random.normal(20, 40, 10)},
        ...
        ...     "site2": {"LSTM": np.random.normal(20, 40, 10),
        ...                 "CNN": np.random.normal(20, 40, 10),
        ...                 "TCN": np.random.normal(20, 40, 10),
        ...                 "CNN-LSTM": np.random.normal(20, 40, 10)},
        ...
        ...     "site3": {"LSTM": np.random.normal(20, 40, 10),
        ...                 "CNN": np.random.normal(20, 40, 10),
        ...                 "TCN": np.random.normal(20, 40, 10),
        ...                 "CNN-LSTM": np.random.normal(20, 40, 10)},
        ...
        ...     "site4": {"LSTM": np.random.normal(20, 40, 10),
        ...                 "CNN": np.random.normal(20, 40, 10),
        ...                 "TCN": np.random.normal(20, 40, 10),
        ...                 "CNN-LSTM": np.random.normal(20, 40, 10)},
        ... }
        ... # define positions of subplots
        >>> rects = dict(site1=221, site2=222, site3=223, site4=224)
        ...
        >>> taylor_plot(observations=observations,
        ...             simulations=simulations,
        ...             axis_locs=rects,
        ...             plot_bias=True,
        ...             cont_kws={'colors': 'blue', 'linewidths': 1.0, 'linestyles': 'dotted'},
        ...             grid_kws={'axis': 'x', 'color': 'g', 'lw': 1.0},
        ...             title="mutiple subplots")
        ...

        Sometimes we don't have actual observation and simulation values as arrays. We can
        still make Taylor plot by providing only standard deviation of observations
        and simulations and coefficient of correlation (R) values
        between simulations and observations.

        >>> observations = {'std': 4.916}
        >>> predictions = {   # pbias is optional
        ...         'Model 1': {'std': 2.80068, 'corr_coeff': 0.49172, 'pbias': -8.85},
        ...         'Model 2': {'std': 3.47, 'corr_coeff': 0.67, 'pbias': -19.76},
        ...         'Model 3': {'std': 3.53, 'corr_coeff': 0.596, 'pbias': 7.81},
        ...         'Model 4': {'std': 2.36, 'corr_coeff': 0.27, 'pbias': -22.78},
        ...         'Model 5': {'std': 2.97, 'corr_coeff': 0.452, 'pbias': -7.99}}
        ...
        >>> taylor_plot(observations,
        ...     predictions,
        ...     title="with statistical parameters")
        ...
        ... # with customized markers
        >>> np.random.seed(313)
        >>> observations =  np.random.normal(20, 40, 10)
        >>> simulations =  {"LSTM": np.random.normal(20, 40, 10),
        ...                 "CNN": np.random.normal(20, 40, 10),
        ...                 "TCN": np.random.normal(20, 40, 10),
        ...                 "CNN-LSTM": np.random.normal(20, 40, 10)}
        >>> taylor_plot(observations=observations,
        ...             simulations=simulations,
        ...             title="customized markers",
        ...             marker_kws={'markersize': 10, 'markeredgewidth': 1.5,
        ...                 'markeredgecolor': 'black'})
        ...
        ... # with customizing bbox
        >>> np.random.seed(313)
        >>> observations =  np.random.normal(20, 40, 10)
        >>> simus =  {"LSTMBasedRegressionModel": np.random.normal(20, 40, 10),
        ...         "CNNBasedRegressionModel": np.random.normal(20, 40, 10),
        ...         "TCNBasedRegressionModel": np.random.normal(20, 40, 10),
        ...         "CNN-LSTMBasedRegressionModel": np.random.normal(20, 40, 10)}
        >>> taylor_plot(observations=observations,
        ...             simulations=simus,
        ...             title="custom_legend",
        ...             leg_kws={'facecolor': 'white',
        ...                 'edgecolor': 'black','bbox_to_anchor':(1.1, 1.05)})

    See :ref:`sphx_glr_auto_examples_taylor_plot.py` for more examples

    .. _Taylor:
        https://doi.org/10.1029/2000JD900719

    .. _contours:
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.contour.html

    .. _axes.grid:
        https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.grid.html

    .. _axes.legend:
        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.legend.html

    """
    # todo replicate https://bookdown.org/david_carslaw/openair/sec-TaylorDiagram.html
    values = False

    if isinstance(observations, np.ndarray):
        observations = {'Reference': observations}
        simulations = {'Reference': simulations}
    elif is_dataframe(observations) or is_series(observations):
        observations = {'Reference': observations.values}
        simulations = {'Reference': simulations}

    # if observations is a dictionary with statistics
    if isinstance(observations, dict) and len(observations) == 1 and 'std' in observations.keys():
        observations = {'Reference': observations}
        simulations = {'Reference': simulations}

    if all([isinstance(v, dict) for v in observations.values()]):  # we are given std, corr_coeff values
        values = True
    else:  # we are given actual arrays
        assert all([isinstance(np.array(v), np.ndarray) for v in observations.values()])
        assert all([len(array) > 1 for array in observations.values()]), f"""
        one or more array in true values has less than 2 values
        {pprint.pprint({key: len(array) for key, array in observations.items()}, width=20)}"""
    scenarios = observations.keys()

    figsizes = {
        1: (8, 6),
        2: (11, 9),
        3: (11, 9),
        4: (12, 10)
    }

    n_plots = len(observations)
    assert n_plots == len(simulations)

    add_ith_interval = kwargs.get('add_idth_interval', False)
    ref_color = kwargs.get('ref_color', 'r')
    intervals = kwargs.get('intervals', [])
    colors = kwargs.get('colors', COLORS)
    extend = kwargs.get('extend', False)
    plot_bias = kwargs.get('plot_bias', False)
    title = kwargs.get('title', "")
    figsize = kwargs.get("figsize", figsizes[n_plots])  # widht and heigt respectively
    sim_marker = kwargs.get("sim_marker", None)
    true_label = kwargs.get("true_label", "Reference")
    show = kwargs.get("show", True)
    corr_alias = kwargs.get("corr_alias", 'Correlation')
    std_alias = kwargs.get("std_alias", 'Standard Deviation')

    if axis_locs is None:
        axis_locs = {k: v for k, v in zip(scenarios, RECTS[len(scenarios)])}

    # importing it at the start will be too early, because every submodule imports from utils
    # so utils must not import something from other submodules at the start

    sims = list(simulations.values())
    models = len(sims[0])

    for m in sims:
        assert len(m) == models

    def msg(key, where="simulations"):
        return f"Scenario {key} does not match any of the provided scenarios in {where}"

    for scen in scenarios:
        if scen not in simulations:
            raise KeyError(msg(scen))
        if scen not in axis_locs:
            raise KeyError(msg(scen, "axis_locs"))

    def get_marker(third_value, _name):
        marker = 'o'

        if sim_marker is not None:
            if isinstance(sim_marker, str):
                return sim_marker
            elif isinstance(sim_marker, dict):
                return sim_marker[_name]

        if third_value:

            if third_value >= 0.0:
                marker = "^"
            else:
                marker = "v"

        return marker

    def get_marker_kws(index):
        if marker_kws:
            return marker_kws

        return {'ms': 10, 'ls': '', 'mfc': colors[index], 'mec': colors[index], 'mew': 1.0}

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=18)

    if axis_fontdict is None:
        axis_fontdict = {'left': {},
                         'right': {},
                         'bottom': {},
                         'top': {}}
    else:
        assert isinstance(axis_fontdict, dict)
        for k in ['left', 'right', 'bottom', 'top']:
            if k not in axis_fontdict:
                axis_fontdict[k] = {}

    add_grid = True
    grid_kws = grid_kws or {}

    if grid_kws is False:
        add_grid = False

    for season in scenarios:

        dia = TaylorDiagram(observations[season],
                            fig=fig,
                            rect=axis_locs[season],
                            label=true_label,
                            axis_fontdict=axis_fontdict,
                            extend=extend,
                            name=season,
                            corr_alias=corr_alias,
                            std_alias=std_alias,
                            )

        dia.samplePoints[0].set_color(ref_color)  # reference point color

        if add_ith_interval:
            for interval in intervals:
                dia.ax.plot(*interval, color='k')

        # Add samples to Taylor diagram
        idx = 0
        for model_name, model in simulations[season].items():
            if values:
                assert isinstance(model, dict)
                stddev = model['std']
                corrcoef = model['corr_coeff']
                third_val = None

                if len(model) > 2:
                    _ = list(model.keys())[2]
                    third_val = list(model.values())[2]
            else:
                sim = np.array(model)
                obs = np.array(observations[season])
                assert sim.size == len(sim)
                assert obs.size == len(obs)
                stddev = np.std(sim)
                corrcoef = corr_coeff(obs.reshape(-1, ), sim.reshape(-1, ))
                third_val = None
                if plot_bias:
                    third_val = pbias(obs, sim)

            marker = get_marker(third_val, model_name)

            dia.add_sample(stddev, corrcoef,
                           marker=marker,
                           **get_marker_kws(idx),
                           label=model_name)

            idx += 1

        if cont_kws is None:
            cont_kws = dict()

        # Add RMS contours, and label them
        contours = dia.add_contours(levels=cont_kws.get('level', 5), colors=cont_kws.get('colors', '0.5'),
                                    linewidths=cont_kws.get('linewidths', 1.5),
                                    linestyles=cont_kws.get('linestyles', None))  # 5 levels
        dia.ax.clabel(contours, inline=cont_kws.get('inline', 1), fontsize=cont_kws.get('label_fs', 10), fmt='%.1f')

        if add_grid:
            dia.add_grid(**grid_kws)  # Add grid
            dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

        if axis_kws is None:
            axis_kws = dict()
        title_fontsize = axis_kws.pop('title_fontsize') if 'title_fontsize' in axis_kws else 14
        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        if len(scenarios) > 1:
            dia._ax.set_title(season.capitalize(), fontdict={'fontsize': title_fontsize})

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html

    if leg_kws is None:
        position = "center" if len(scenarios) == 4 else "upper right"
        leg_kws = {'loc': position}

    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               **leg_kws)

    if show:
        plt.show()

    return fig
