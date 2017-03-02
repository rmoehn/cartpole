import collections
import itertools

import matplotlib
matplotlib.use('GTK3Agg')
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

def norm_cum_hist(xs, bins):
    all_xs      = xs.compressed()
    n_xs        = all_xs.shape[0]
    hist, x_edges, y_edges = np.histogram2d(all_xs, np.arange(n_xs),
                                            bins=[bins, 25],
                                            range=[[-2.4, 2.4],
                                                   [0, n_xs]])
    hist = hist.T
    cum_hist = np.cumsum(hist, axis=0)
    return cum_hist / np.sum(cum_hist, axis=1)[:,None], x_edges, y_edges


# Plot in procedure pattern credits:
# https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def plot_xss_cum_hist_devel(xs, ax=None, bins=25):
    nch, x_edges, y_edges = norm_cum_hist(xs, bins)

    if ax is None:
        ax = plt.gca()

    X, Y = np.meshgrid(x_edges, y_edges)
    mesh = ax.pcolormesh(X, Y, nch)

    return mesh


def plot_xss_cum_hist_change(xs, ax=None, bins=25):
    nch, _, _ = norm_cum_hist(xs, bins)

    diff = np.abs(np.diff(nch, axis=0))
    nozero_nch = np.where(nch == 0, 100, nch)
    rel_diff = diff / nozero_nch[1:] * 100
    changes = np.max(rel_diff, axis=1)

    if ax is None:
        ax = plt.gca()

    ax.plot(changes)


# Credits: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.fill
def plot_mean_std_change(xs, ax=None, label=None):
    all_xs = xs.compressed()
    n_xs   = all_xs.shape[0]

    cum_mean = [np.mean(all_xs[:i]) for i in xrange(1, n_xs, n_xs // 100)]
    cum_mean.append(np.mean(all_xs))
    cum_mean = np.array(cum_mean)

    cum_std = [np.std(all_xs[:i]) for i in xrange(1, n_xs, n_xs // 100)]
    cum_std.append(np.std(all_xs))
    cum_std = np.array(cum_std)

    ax = ax or plt.gca()
    p = ax.plot(cum_mean, xrange(cum_mean.shape[0]), label=label)
    # Credits: http://stackoverflow.com/a/36700159/5091738
    ax.fill_betweenx(xrange(cum_mean.shape[0]),
            cum_mean - cum_std, cum_mean + cum_std, alpha=0.3,
            facecolor=p[0].get_color())



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


def plot_xs_hist(xs, ax, bins=25, label=""):
    ax.hist(xs, range=(-2.4, 2.4), bins=bins, normed=True, alpha=0.3,
            label=label)


Axes = collections.namedtuple(
            'Axes', ['el', 'devel', 'devel2', 'comp', 'comp2'])

def arrange_algo_full():
    fig = plt.figure(figsize=(10, 12))

    gs = gridspec.GridSpec(4, 8)

    unintint = ("uninterrupted", "interrupted")

    ax = Axes(*([None, None] for _ in xrange(len(Axes._fields))))

    for (i, n) in enumerate(unintint):
        ax.el[i] = fig.add_subplot(gs[i, 0:3])
        ax.el[i].set_title("episode lengths")
        ax.el[i].set_xlabel("episode nr.")
        ax.el[i].set_ylabel("duration/total reward")

        ax.devel[i] = fig.add_subplot(gs[i, 3:6])
        ax.devel[i].set_title("cumulative hist. over time")
        ax.devel[i].set_xlabel("x-coordinate of cart")
        ax.devel[i].set_ylabel("timestep nr.")

        ax.devel2[i] = fig.add_subplot(gs[i, 6:])
        ax.devel2[i].set_xlabel("x-coordinate of cart")
        ax.devel2[i].set_ylabel("timestep nr.")

        ax.comp[i] = fig.add_subplot(gs[2+i, 0:4])
        ax.comp[i].set_title("histogram over all timesteps before 1.0 crosses")
        ax.comp[i].set_xlabel("x-coordinate of cart")
        ax.comp[i].set_ylabel("proportion of time spent")

        ax.comp2[i] = fig.add_subplot(gs[2+i, 4:8])
        ax.comp2[i].set_title("histogram over all timesteps before 1.0 crosses")
        ax.comp2[i].set_xlabel("x-coordinate of cart")
        ax.comp2[i].set_ylabel("proportion of time spent")

    fig.tight_layout()

    return fig, ax
