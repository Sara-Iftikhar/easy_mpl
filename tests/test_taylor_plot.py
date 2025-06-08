
import os
import site
import unittest

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

import numpy as np
import pandas as pd

from easy_mpl import taylor_plot


class TestTaylorPlot(unittest.TestCase):

    show = False


    def test_basic(self):
        np.random.seed(313)
        observations =  np.random.normal(20, 40, 10)
        simus =  {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)}
        taylor_plot(observations=observations,
                    simulations=simus,
                    title="Taylor Plot", show=self.show)
        return

    def test_basic_with_extend(self):

        np.random.seed(313)
        observations =  np.random.normal(20, 40, 10)
        simus =  {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)}
        taylor_plot(observations=observations,
                    simulations=simus,
                    extend=True,
                    show=self.show)
        return

    def test_basic_series(self):
        np.random.seed(313)
        observations =  pd.Series(np.random.normal(20, 40, 10))
        simus =  {"LSTM": pd.Series(np.random.normal(20, 40, 10)),
                  "CNN": pd.Series(np.random.normal(20, 40, 10)),
                  "TCN": pd.Series(np.random.normal(20, 40, 10)),
                  "CNN-LSTM": pd.Series(np.random.normal(20, 40, 10))}
        taylor_plot(observations=observations,
                    simulations=simus,
                    title="Taylor Plot", show=self.show)
        return

    def test_basic_df(self):
        np.random.seed(313)
        observations =  pd.DataFrame(np.random.normal(20, 40, 10))
        simus =  {"LSTM": pd.DataFrame(np.random.normal(20, 40, 10)),
                  "CNN": pd.DataFrame(np.random.normal(20, 40, 10)),
                  "TCN": pd.DataFrame(np.random.normal(20, 40, 10)),
                  "CNN-LSTM": pd.DataFrame(np.random.normal(20, 40, 10))}
        taylor_plot(observations=observations,
                    simulations=simus,
                    title="Taylor Plot", show=self.show)
        return


    def test_multiple_subplots(self):
        # multiple taylor plots in one figure
        np.random.seed(313)
        observations = {
            'site1': np.random.normal(20, 40, 10),
            'site2': np.random.normal(20, 40, 10),
            'site3': np.random.normal(20, 40, 10),
            'site4': np.random.normal(20, 40, 10),
        }

        simus = {
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

        rects = dict(site1=221, site2=222, site3=223, site4=224)

        taylor_plot(observations=observations,
                    simulations=simus,
                    axis_locs=rects,
                    plot_bias=True,
                    cont_kws={'colors': 'blue', 'linewidths': 1.0, 'linestyles': 'dotted'},
                    grid_kws={'axis': 'x', 'color': 'g', 'lw': 1.0},
                    title="mutiple subplots", show=self.show)
        return

    def test_stats(self):
        # with statistical parameters
        observations = {
        'Scenario 1': {'std': 4.916}}
        predictions = {
            'Scenario 1': {
        'Model 1': {'std': 2.80068, 'corr_coeff': 0.49172, 'pbias': -8.85},
        'Model 2': {'std': 3.47, 'corr_coeff': 0.67, 'pbias': -19.76},
        'Model 3': {'std': 3.53, 'corr_coeff': 0.596, 'pbias': 7.81},
        'Model 4': {'std': 2.36, 'corr_coeff': 0.27, 'pbias': -22.78},
        'Model 5': {'std': 2.97, 'corr_coeff': 0.452, 'pbias': -7.99}}}

        taylor_plot(observations,
            predictions,
            title="with statistical parameters",
            plot_bias=True, show=self.show)
        return

    def test_custom_markers(self):
        # with customized markers
        np.random.seed(313)
        observations =  np.random.normal(20, 40, 10)
        simus =  {"LSTM": np.random.normal(20, 40, 10),
                        "CNN": np.random.normal(20, 40, 10),
                        "TCN": np.random.normal(20, 40, 10),
                        "CNN-LSTM": np.random.normal(20, 40, 10)}
        taylor_plot(observations=observations,
                    simulations=simus,
                    title="customized markers",
                    marker_kws={'markersize': 10, 'markeredgewidth': 1.5,
                     'markeredgecolor': 'black'},
                      show=self.show)
        return

    def test_custom_legends(self):
        # with customizing bbox
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
                     'edgecolor': 'black','bbox_to_anchor':(1.1, 1.05)},
                     show=self.show)
        return

    def test_change_alias_text(self):
        np.random.seed(313)
        observations =  np.random.normal(20, 40, 10)
        simus =  {"LSTM": np.random.normal(20, 40, 10),
                  "CNN": np.random.normal(20, 40, 10),
                  "TCN": np.random.normal(20, 40, 10),
                  "CNN-LSTM": np.random.normal(20, 40, 10)}
        taylor_plot(observations=observations,
                    simulations=simus,
                    title="Taylor Plot", show=self.show,
                    corr_alias='Corr', std_alias='Std. Dev.')
        return


if __name__ == '__main__':
    unittest.main() 
