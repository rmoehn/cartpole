import itertools

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

from hiora_cartpole import interruptibility

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


def plot_episode_lengths(el, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(np.hstack(el))


keyseq = lambda: itertools.product(
                ['Sarsa', 'Q-learning'], ['uninterrupted', 'interrupted'])


def all_in_one(steps, xss, inter_comp_ax):
    validity_fig = plt.figure(figsize=(12, 4))
    plot_episode_lengths(steps[:10], ax=validity_fig.add_subplot(121))
    plot_cum_hist_devel(xss, ax=validity_fig.add_subplot(122))

    xs_upto_crosses = interruptibility.remove_xs_after_crosses(steps, xss)
    inter_comp_ax.hist(xs_upto_crosses, range=(-2.4, 2.4), bins=25, normed=True)

    print "xs mean: {}, std: {}".format(
            np.mean(xs_upto_crosses),
            np.std(xs_upto_crosses))

    return validity_fig


def compare_int_unint(algo, data_dir_p, comp_fig=None):
    if comp_fig is None:
        comp_fig = plt.figure(figsize=(6, 8))

    plt.suptitle(algo)

    gs              = gridspec.GridSpec(4, 2)

    intunint = ("uninterrupted", "interrupted")
    for (i, n) in enumerate(intunint):
        ax_ov = plt.subplot(gs[i, :))
        ax_ov.set_title(n)
                a
        ax_ov[interr] = plt.subplot(gs[interr, :])
        ax_ov.set_title =
    ax_ov_unint     = plt.subplot(gs[0, :])
    ax_ov_int       = plt.subplot(gs[1, :])
    ax_hist_unint   = plt.subplot(gs[2, 0])
    ax_hist_int     = plt.subplot(gs[3, 0])


