
import os
import site

package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
site.addsitedir(package_path)

from easy_mpl import circular_bar_plot

vals = [0.33, 0.35, 0.12, 0.20, 0.18]
labels = ['a', 'b', 'c', 'd', 'e']

circular_bar_plot(vals, labels, sort=True)