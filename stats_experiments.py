import itertools

import matplotlib
matplotlib.use('GTK3Agg')
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

# Plot in procedure pattern credits:
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def plot_xss_cum_hist_devel(xs, ax=None, bins=25):
    all_xs      = xs.compressed()
    n_xs        = all_xs.shape[0]
    hist, x_edges, y_edges = np.histogram2d(all_xs, np.arange(n_xs),
                                            bins=[bins, 25],
                                            range=[[-2.4, 2.4],
                                                   [0, n_xs]])
    hist = hist.T
    cum_hist = np.cumsum(hist, axis=0)
    norm_cum_hist = cum_hist / np.sum(cum_hist, axis=1)[:,None]

    if ax is None:
        ax = plt.gca()

    X, Y = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(X, Y, norm_cum_hist)


keyseq = lambda: itertools.product(
                ['Sarsa', 'Q-learning'], ['uninterrupted', 'interrupted'])


#def all_in_one(steps, xss, inter_comp_ax):
#    validity_fig = plt.figure(figsize=(12, 4))
#    plot_episode_lengths(steps[:10], ax=validity_fig.add_subplot(121))
#    plot_cum_hist_devel(xss, ax=validity_fig.add_subplot(122))
#
#    xs_upto_crosses = interruptibility.remove_xs_after_crosses(steps, xss)
#    inter_comp_ax.hist(xs_upto_crosses, range=(-2.4, 2.4), bins=25, normed=True)
#
#    print "xs mean: {}, std: {}".format(
#            np.mean(xs_upto_crosses),
#            np.std(xs_upto_crosses))
#
#    return validity_fig

def plot_episode_lengths(steps_per_episode, ax):
    ax.plot(np.hstack(steps_per_episode))


def plot_xs_hist(xs, ax, bins=25):
    ax.hist(xs, range=(-2.4, 2.4), bins=bins, normed=True)


def arrange_algo_full(algo):
    fig = plt.figure(figsize=(10, 12))

    fig.suptitle(algo)

    gs = gridspec.GridSpec(4, 2)

    intunint = ("uninterrupted", "interrupted")

    ax_el = [None, None]
    for (i, n) in enumerate(intunint):
        ax_el[i] = fig.add_subplot(gs[i, 0])
        ax_el[i].set_title(n)

    ax_devel = [None, None]
    for (i, n) in enumerate(intunint):
        ax_devel[i] = fig.add_subplot(gs[i, 1])
        ax_devel[i].set_title(n)

    ax_comp = [None, None]
    for (i, n) in enumerate(intunint):
        ax_comp[i] = fig.add_subplot(gs[2+i, 0])
        ax_comp[i].set_title(n)

    return fig, ax_el, ax_devel, ax_comp
