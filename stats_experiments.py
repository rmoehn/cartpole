import itertools

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib import pyplot as plt
import numpy as np

# Plot in procedure pattern credits:
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def plot_cum_hist_devel(xss, ax=None):
    all_xs = np.hstack(xss)
    run_of_x = np.hstack([np.full(len(xs), n_run)
                        for n_run, xs in enumerate(xss)])
    hist, x_edges, y_edges = np.histogram2d(all_xs, run_of_x,
                                            bins=[51, len(xss) // 4])
    hist = hist.T
    cum_hist = np.cumsum(hist, axis=0)
    norm_cum_hist = cum_hist / np.sum(cum_hist, axis=1)[:,None]

    if ax is None:
        ax = plt.gca()

    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, norm_cum_hist)

keyseq = lambda: itertools.product(
                ['Sarsa', 'Q-learning'], ['uninterrupted', 'interrupted'])
