
__all__ = ["parallel_coordinates"]


from typing import Union, Any

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from .utils import _rescale


def parallel_coordinates(
        data: Union[np.ndarray, Any],
        categories: Union[np.ndarray, list] = None,
        names: list = None,
        cmap: str = None,
        linestyle: str = "bezier",
        coord_title_kws: dict = None,
        title: str = 'Parallel Coordinates Plot',
        figsize: tuple = None,
        ticklabel_kws: dict = None,
        show: bool = True
) -> plt.Axes:
    """
    parallel coordinates plot
    modifying after https://stackoverflow.com/a/60401570/5982232

    Parameters
    ----------
        data : array, DataFrame
            a two dimensional array with the shape (rows, columns). It can also
            be pandas DataFrame
        categories : list, array
            1 dimensional array which contain class labels of the of each row in
            data. It can be either categorical or continuous numerical values.
            If not given, colorbar will not be drawn. The length of categroes
            array must be equal to length of/rows in data.
        names : list, optional
            Labels for columns in data. It's length should be equal to number of
            oclumns in data.
        cmap : str, optional
            colormap to be used
        coord_title_kws : dict, optional
            keyword arguments for coodinate titles. All of these arguments will go to
            :obj:`matplotlib.axes.Axes.set_xticklabels`
        linestyle : str, optional
            either "straight" or "bezier". Default is "bezier".
        title : str, optional
            title for the Figure
        figsize : tuple, optional
            figure size
        ticklabel_kws : dict, optional
            keyword arguments for ticklabels on y-axis
        show : bool, optional
            whether to show the plot or not

    Returns
    -------
        matplotlib Axes

    Examples
    --------
    >>> import random
    >>> import numpy as np
    >>> import pandas as pd
    >>> from easy_mpl import parallel_coordinates
    ...
    >>> ynames = ['P1', 'P2', 'P3', 'P4', 'P5']  # feature/column names
    >>> N1, N2, N3 = 10, 5, 8
    >>> N = N1 + N2 + N3
    >>> categories_ = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> y1 = np.random.uniform(0, 10, N) + 7
    >>> y2 = np.sin(np.random.uniform(0, np.pi, N))
    >>> y3 = np.random.binomial(300, 1 / 10, N)
    >>> y4 = np.random.binomial(200, 1 / 3, N)
    >>> y5 = np.random.uniform(0, 800, N)
    ... # combine all arrays into a pandas DataFrame
    >>> data_np = np.column_stack((y1, y2, y3, y4, y5))
    >>> data_df = pd.DataFrame(data_np, columns=ynames)
    ... # using a DataFrame to draw parallel coordinates
    >>> parallel_coordinates(data_df, names=ynames)
    ... # using continuous values for categories
    >>> parallel_coordinates(data_df, names=ynames, categories=np.random.randint(0, 5, N))
    ... # using categorical classes
    >>> parallel_coordinates(data_df, names=ynames, categories=random.choices(categories_, k=N))
    ... # using numpy array instead of DataFrame
    >>> parallel_coordinates(data_df.values, names=ynames)
    ... # with customized tick labels
    >>> parallel_coordinates(data_df.values, ticklabel_kws={"fontsize": 8, "color": "red"})
    ... # using straight lines instead of bezier
    >>> parallel_coordinates(data_df, linestyle="straight")
    ... # with categorical class labels
    >>> data_df['P5'] = random.choices(categories_, k=N)
    >>> parallel_coordinates(data_df, names=ynames)
    ... # with categorical class labels and customized ticklabels
    >>> data_df['P5'] = random.choices(categories_, k=N)
    >>> parallel_coordinates(data_df,  ticklabel_kws={"fontsize": 8, "color": "red"})

    See :ref:`sphx_glr_auto_examples_parallel_coordinates.py` for more examples

    Note
    ----
        If nans are present in data or categories, all the corresponding enteries/rows
        will be removed.
    """

    try:
        import pandas as pd
    except (ModuleNotFoundError, ImportError):
        raise NotImplemented(f"You must install pandas to draw parallel plot")

    if cmap is None:
        cmap = "Blues"

    if isinstance(data, np.ndarray):
        assert data.ndim == 2, f"{data.ndim} dimensional data not allowed. It must be 2d"
        if names is None:
            names = [f"Feat_{i}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=names)

    if hasattr(data, "columns"):
        names = names or data.columns.tolist()

    if len(names) != data.shape[1]:
        raise ValueError(f"""
            provided names have length {len(names)} but data has {data.shape[1]} columns""")

    show_colorbar = True
    if categories is None:
        show_colorbar = False
        categories = np.linspace(0, 1, len(data))

    categories = np.array(categories)
    assert len(categories) == len(data)

    # remove NaN values based upon nan values in data
    if data.isna().sum().sum() > 0:
        df_nan_idx = data.isna().any(axis=1)
        categories = categories[~df_nan_idx]
        data = data[~df_nan_idx]

    _is_categorical = False
    cat_encoded = categories
    if not np.issubdtype(categories.dtype, np.number):
        # category contains categorical/non-numeri values
        cat_encoded = label_encoder(categories)
        _is_categorical = True

    if not _is_categorical:  # because we can't do np.isnan for categorical values
        # if there are still any nans in categories, remove them
        cat_nan_idx = np.isnan(categories)
        if cat_nan_idx.any():
            categories = categories[~cat_nan_idx]
            data = data[~cat_nan_idx]

    num_cols = data.shape[1]
    num_lines = len(data)

    # find out which columns are categorical and which are numerical
    enc_data = data.copy()
    cols = {}
    for idx, col in enumerate(data.columns):
        _col = data[col]
        if is_categorical(data[col].values):
            col_encoded = label_encoder(data[col].values)
            cols[idx] = {'cat': True, 'original': _col}
            enc_data[col] = col_encoded
        else:
            cols[idx] = {'cat': False}

    # organize the data
    enc_data = enc_data.astype(float)
    ymins = np.min(enc_data.values, axis=0)  # ys.min(axis=0)
    ymaxs = np.max(enc_data.values, axis=0)  # ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(enc_data.values)
    zs[:, 0] = enc_data.iloc[:, 0]
    zs[:, 1:] = (enc_data.iloc[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=figsize)

    axes = [host] + [host.twinx() for _ in range(num_cols - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (num_cols - 1)))

        if cols[i]['cat']:
            categories = np.unique(cols[i]['original'])
            new_ticks = np.unique(enc_data.iloc[:, i]).astype("float32")
            ax.set_yticks(new_ticks)
            ax.set_yticklabels(categories)

        if ticklabel_kws:
            if cols[i]['cat']:
                ticks_loc = [l._text for l in ax.get_yticklabels()]
            else:
                ticks_loc = ax.get_yticks().tolist()

            ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels([label_format(x) for x in ticks_loc], **ticklabel_kws)

    if coord_title_kws is None:
        coord_title_kws = dict()

    _coord_title_kws = {'fontsize': 14}

    _coord_title_kws.update(coord_title_kws)

    host.set_xlim(0, num_cols - 1)
    host.set_xticks(range(num_cols))
    host.set_xticklabels(names, **_coord_title_kws)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)

    if title:
        host.set_title(title, fontsize=18)

    # category between 0.2,1 to map colors to their values
    cat_norm = _rescale(cat_encoded, 0.2)

    for j in range(num_lines):
        # color of each line is based upon corresponding value in category
        colors = getattr(cm, cmap)(cat_norm[j])

        if linestyle == "straight":
            # to just draw straight lines between the axes:
            host.plot(range(num_cols), zs[j, :], c=colors)
        else:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            x_coords = [x for x in np.linspace(0, len(data) - 1, len(data) * 3 - 2, endpoint=True)]
            y_coords = np.repeat(zs[j, :], 3)[1:-1]
            verts = list(zip(x_coords, y_coords))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors)
            host.add_patch(patch)

    if show_colorbar:
        norm = cm.colors.Normalize(np.min(cat_encoded), np.max(cat_encoded))

        cb = cm.ScalarMappable(norm, cmap=cmap)

        if _is_categorical:
            cbar = fig.colorbar(cb, orientation="vertical", pad=0.1, ax=ax)
            ticks = cbar.get_ticks()
            new_ticks = np.linspace(ticks[0], ticks[-1], len(np.unique(categories)))
            cbar.set_ticks(new_ticks)
            cbar.set_ticklabels(np.unique(categories))

        else:
            cbar = fig.colorbar(cb, orientation="vertical", pad=0.1, ax=ax)

        cax = cbar.ax  # todo
        # Turn spines off and create white grid.
        if isinstance(cax.spines, dict):
            for sp in cax.spines:
                cax.spines[sp].set_visible(False)
        else:
            cax.spines[:].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return host


def label_format(x):
    if isinstance(x, float):
        return round(x, 3)
    else:
        return x


def is_categorical(array) -> bool:
    return not np.issubdtype(array.dtype, np.number)


def label_encoder(arr):
    # label encoder of numpy array with categorical values
    return np.unique(arr, return_inverse=True)[1]
