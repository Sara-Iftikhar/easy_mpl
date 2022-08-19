"""
===============
o. taylor plot
===============

A Taylor plot can be used to show statistical summary of one or more measurements/models.

"""

import numpy as np
from easy_mpl import taylor_plot

# sphinx_gallery_thumbnail_number = -1

#############################

np.random.seed(313)
observations =  np.random.normal(20, 40, 10)
simulations =  {"LSTM": np.random.normal(20, 40, 10),
            "CNN": np.random.normal(20, 40, 10),
            "TCN": np.random.normal(20, 40, 10),
            "CNN-LSTM": np.random.normal(20, 40, 10)}
taylor_plot(observations=observations,
            simulations=simulations,
            title="Taylor Plot")

#############################

# multiple taylor plots in one figure
#-----------------------

np.random.seed(313)
observations = {
    'site1': np.random.normal(20, 40, 10),
    'site2': np.random.normal(20, 40, 10),
    'site3': np.random.normal(20, 40, 10),
    'site4': np.random.normal(20, 40, 10),
}

simulations = {
    "site1": {"LSTM": np.random.normal(20, 40, 10),
                "CNN": np.random.normal(20, 40, 10),
                "TCN": np.random.normal(20, 40, 10),
                "CNN-LSTM": np.random.normal(20, 40, 10)},

    "site2": {"LSTM": np.random.normal(20, 40, 10),
                "CNN": np.random.normal(20, 40, 10),
                "TCN": np.random.normal(20, 40, 10),
                "CNN-LSTM": np.random.normal(20, 40, 10)},

    "site3": {"LSTM": np.random.normal(20, 40, 10),
                "CNN": np.random.normal(20, 40, 10),
                "TCN": np.random.normal(20, 40, 10),
                "CNN-LSTM": np.random.normal(20, 40, 10)},

    "site4": {"LSTM": np.random.normal(20, 40, 10),
                "CNN": np.random.normal(20, 40, 10),
                "TCN": np.random.normal(20, 40, 10),
                "CNN-LSTM": np.random.normal(20, 40, 10)},
}

# define positions of subplots

rects = dict(site1=221, site2=222, site3=223, site4=224)

taylor_plot(observations=observations,
            simulations=simulations,
            axis_locs=rects,
            plot_bias=True,
            cont_kws={'colors': 'blue', 'linewidths': 1.0, 'linestyles': 'dotted'},
            grid_kws={'axis': 'x', 'color': 'g', 'lw': 1.0},
            title="mutiple subplots")

#############################
# using statistics instead of arrays

observations = {'std': 4.916}
predictions = {   # pbias is optional
        'Model 1': {'std': 2.80068, 'corr_coeff': 0.49172, 'pbias': -8.85},
        'Model 2': {'std': 3.47, 'corr_coeff': 0.67, 'pbias': -19.76},
        'Model 3': {'std': 3.53, 'corr_coeff': 0.596, 'pbias': 7.81},
        'Model 4': {'std': 2.36, 'corr_coeff': 0.27, 'pbias': -22.78},
        'Model 5': {'std': 2.97, 'corr_coeff': 0.452, 'pbias': -7.99}}



taylor_plot(observations,
    predictions,
    title="with statistical parameters")

#############################

# with customized markers
#-----------------------

np.random.seed(313)
observations =  np.random.normal(20, 40, 10)
simulations =  {"LSTM": np.random.normal(20, 40, 10),
                "CNN": np.random.normal(20, 40, 10),
                "TCN": np.random.normal(20, 40, 10),
                "CNN-LSTM": np.random.normal(20, 40, 10)}
taylor_plot(observations=observations,
            simulations=simulations,
            title="customized markers",
            marker_kws={'markersize': 10, 'markeredgewidth': 1.5,
                'markeredgecolor': 'black'})

#############################

# with customizing bbox
#-----------------------

np.random.seed(313)
observations =  np.random.normal(20, 40, 10)
simus =  {"LSTMBasedRegressionModel": np.random.normal(20, 40, 10),
        "CNNBasedRegressionModel": np.random.normal(20, 40, 10),
        "TCNBasedRegressionModel": np.random.normal(20, 40, 10),
        "CNN-LSTMBasedRegressionModel": np.random.normal(20, 40, 10)}
taylor_plot(observations=observations,
            simulations=simus,
            title="custom_legend",
            leg_kws={'facecolor': 'white',
                'edgecolor': 'black','bbox_to_anchor':(1.1, 1.05)})