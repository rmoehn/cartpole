import collections
import itertools

import matplotlib
matplotlib.use('GTK3Agg')
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np

from hiora_cartpole import interruptibility
import saveloaddata

# Credits: itertools documentation.
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


# Credits: https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
def kld(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def jsd(p, q): # Jenses-Shannon divergence
    m = 0.5 * (p + q)

    # http://stackoverflow.com/a/29950752/5091738
    # Division by zero can only occur in kld when an entry of m is zero. This
    # can only happen if both p and q are zero. In this case, kld ignores the
    # nan resulting from log(inf), because its first argument is zero.
    # Therefore, it is is safe to ignore this error.
    with np.errstate(divide='ignore', invalid='ignore'):
        res = 0.5 * (kld(p, m) + kld(q, m))

    return res


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


def plot_jsd_devel(xs, bins=25, ax=None):
    nch, _, _ = norm_cum_hist(xs, bins)

    jsds = (jsd(nch[i], nch[i+1]) for i in xrange(nch.shape[0] - 1))

    ax = ax or plt.gca()
    ax.plot(np.fromiter(jsds, np.float64))


def plot_jsd_comp_final(xs, bins=25, ax=None):
    nch, _, _ = norm_cum_hist(xs, bins)

    jsds = (jsd(nch[i], nch[-1]) for i in xrange(nch.shape[0] - 1))

    ax = ax or plt.gca()
    ax.plot(np.fromiter(jsds, np.float64))


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
            'Axes', ['el', 'devel', 'devel2', 'jsd', 'hist', 'hist2',
                     'meanstd'])

unintint = ("uninterrupted", "interrupted")


def arrange_algo_full():
    fig = plt.figure(figsize=(10, 12))

    gs = gridspec.GridSpec(10, 5)

    ax = Axes(*([None, None] for _ in xrange(len(Axes._fields))))

    for i in xrange(len(unintint)):
        ax.el[i] = fig.add_subplot(gs[5*i:, 0])
        ax.el[i].set_title("episode lengths")
        ax.el[i].set_xlabel("episode nr.")
        ax.el[i].set_ylabel("duration/total reward")

        ax.devel[i] = fig.add_subplot(gs[:4, i+1])
        ax.devel[i].set_title("hist. over time")
        ax.devel[i].set_xlabel("x-coordinate of cart")
        ax.devel[i].set_ylabel("timestep nr.")

        ax.devel2[i] = fig.add_subplot(gs[4:6, i+1])
        ax.devel2[i].set_title("hist. over time")
        ax.devel2[i].set_xlabel("x-coordinate of cart")
        ax.devel2[i].set_ylabel("timestep nr.")

        ax.jsd[i] = fig.add_subplot(gs[6:, i+1])
        ax.jsd[i].set_title("Jensen-Shannon div.")
        ax.jsd[i].set_xlabel("JSD")
        ax.jsd[i].set_ylabel("timestep nr.")
        ax.jsd[i].legend()

    ax = ax._replace(hist=fig.add_subplot(gs[:5, 3]))
    ax.hist.set_title("histogram over all timesteps before 1.0 crosses")
    ax.hist.set_xlabel("x-coordinate of cart")
    ax.hist.set_ylabel("proportion of time spent")
    ax.hist.legend()

    ax = ax._replace(hist2=fig.add_subplot(gs[5:, 3]))
    ax.hist2.set_title("histogram over all timesteps before 1.0 crosses")
    ax.hist2.set_xlabel("x-coordinate of cart")
    ax.hist2.set_ylabel("proportion of time spent")
    ax.hist2.legend()

    ax = ax._replace(meanstd=fig.add_subplot(gs[:, 4]))
    ax.meanstd.set_title("mean and std")
    ax.meanstd.set_xlabel("mean +/- 1 std")
    ax.meanstd.set_ylabel("timestep nr.")
    ax.meanstd.legend()

    fig.tight_layout()

    return fig, ax


def load_plot_all(algo, algo_sub, interr01, ax, data_dir_p):
    with saveloaddata.load_res(algo + algo_sub, unintint[interr01],
            data_dir_p) as res:
        el = res[0]
        xs = interruptibility.rsxs2nparray(*res)

    plot_episode_lengths(el[:10], ax.el[interr01])
    before_cross = interruptibility.mask_after_cross(xs)
    plot_xss_cum_hist_devel(before_cross, ax.devel[interr01], bins=25)
    plot_xss_cum_hist_devel(before_cross, ax.devel2[interr01], bins=2)
    plot_xs_hist(before_cross.compressed(), ax.comp[interr01], bins=25)
    plot_xs_hist(before_cross.compressed(), ax.comp2[interr01], bins=2)

    print "%10s %13s mean: %1.4f std: %1.4f" % \
        (algo, unintint[interr01], np.mean(before_cross), np.std(before_cross))
