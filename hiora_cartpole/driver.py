# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import functools
import itertools
from itertools import imap, islice, izip, dropwhile

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import cm, pyplot
import numpy as np
import pyrsistent

def iterate(f, x):
    while True:
        yield x
        x = f(x)

#### Some iterators

# Credits: https://docs.python.org/2/library/itertools.html

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)


# Timestep that the driver sees
DTimestep = pyrsistent.immutable(
                'observation, reward, done, experience, action',
                name='DTimestep')


# pylint: disable=too-many-arguments
def train(env, learner, experience, n_episodes, max_steps, is_render=False,
        is_wrapup_at_max_steps=True):
    steps_per_episode = np.zeros(n_episodes, dtype=np.int32)
    alpha_per_episode = np.empty(n_episodes)

    for n_episode in xrange(n_episodes):
        observation = env.reset()
        reward      = 0
        done        = False

        t = 0
        for t in xrange(max_steps):
            is_render and env.render() # pylint: disable=expression-not-assigned
            experience, action = learner.think(experience, observation, reward,
                                               done)
            observation, reward, done, _ = env.step(action)

            if done and (t != (max_steps - 1) or is_wrapup_at_max_steps):
                print "Wrapping up"
                print "%3d %3d" % (n_episode, t)
                steps_per_episode[n_episode] = t
                alpha_per_episode[n_episode] = experience.p_alpha
                experience = learner.wrapup(experience, observation, reward)
                break
        else:
            print "%3d %3d" % (n_episode, t)
            steps_per_episode[n_episode] = max_steps
            alpha_per_episode[n_episode] = experience.p_alpha

    return experience, steps_per_episode, alpha_per_episode


def greedy_act(e, o):
    return max(xrange(e.act_space.n),
               key=lambda a: e.feature_vec(o, a).dot(e.theta))


def exec_greedy(env, experience, n_episodes, max_steps, is_render=False):
    steps_per_episode = np.zeros(n_episodes, dtype=np.int32)

    for n_episode in xrange(n_episodes):
        observation = env.reset()
        done        = False
        t           = 0

        for t in xrange(max_steps):
            is_render and env.render() # pylint: disable=expression-not-assigned
            action                  = greedy_act(experience, observation)
            observation, _, done, _ = env.step(action)

            if done:
                break

        steps_per_episode[n_episode] = t

    return steps_per_episode


def make_next_dtimestep(env, think):
    def next_dtimestep_inner(dtimestep):
        if dtimestep.done:
            return dtimestep.set(reward=0, action=None)

        new_experience, new_action = think(dtimestep.experience,
                                           dtimestep.observation,
                                           dtimestep.reward)
        new_observation, new_reward, new_done, _ = env.step(new_action)

        return DTimestep(new_observation, new_reward, new_done,
                         new_experience, new_action)

    return next_dtimestep_inner


def make_train_and_prep(env, next_dtimestep, wrapup):
    def train_and_prep_inner(first_dtimestep, max_steps=-1):
        enumerator  = itertools.count(0) if max_steps < 0 else xrange(max_steps)
        dtimestep   = first_dtimestep
        n_dtimestep = 0
        for n_dtimestep in enumerator:
            dtimestep = next_dtimestep(dtimestep)

            if dtimestep.done:
                break

        experience = wrapup(dtimestep.experience,
                            dtimestep.observation,
                            dtimestep.reward)

        return n_dtimestep + 1, \
               DTimestep(env.reset(), 0, False, experience, None)

    return train_and_prep_inner


def averages(numbers_iter, avg_window):
    return imap(np.average, grouper(numbers_iter, avg_window))


# pylint: disable=too-many-arguments, too-many-locals
def train_until_converged(env, train_and_prep, init_experience,
        max_steps, max_episodes, avg_window, max_diff):
    """
    Train until the rewards roughly stay the same

    This is a crude heuristic and not a real convergence test.
    """

    first_dtimestep   = DTimestep(env.reset(), 0, False, init_experience, None)

    train_and_prep_ms = functools.partial(train_and_prep, max_steps=max_steps)
    cnts_dtimesteps   = iterate(lambda (_, dt): train_and_prep_ms(dt),
                                (0, first_dtimestep))

    numbered_cnts_dtimesteps = izip(itertools.count(0), cnts_dtimesteps)
    cnts                     = imap(lambda (_, (c, d)):
                                    c, numbered_cnts_dtimesteps)
    limited_cnts             = islice(cnts, max_episodes - 1)

    # The underlying iterator is cnts_dtimesteps. The following will implicitly
    # consume cnts_timesteps until the difference between the averages of
    # adjacent groups falls below max_diff. Mutability is crazy, isn't it?
    next(
        dropwhile(lambda avg: avg >= max_diff,
            imap(lambda (a1, a2): abs(a1 - a2),
                pairwise(
                    averages(limited_cnts, avg_window)
                )
            )
        ),
    )

    nr, cnt_dtimestep = next(numbered_cnts_dtimesteps)
    approx_last_avg   = next( averages(cnts, avg_window) )

    return nr - 2 * avg_window, approx_last_avg, cnt_dtimestep[1].experience
        # Subtracting, because it converges, but we only notices after we've
        # compared the averages of the next two groups.


def cnts_dtimesteps_iter(env, train_and_prep, init_experience, max_steps):
    first_dtimestep   = DTimestep(env.reset(), 0, False, init_experience, None)
    train_and_prep_ms = functools.partial(train_and_prep, max_steps=max_steps)

    return iterate(lambda (_, dt): train_and_prep_ms(dt), (0, first_dtimestep))


def train_return_thetas(cnts_dtimesteps, n_episodes):
    _, first_dtimestep = next(cnts_dtimesteps)

    thetas = np.empty((n_episodes, first_dtimestep.experience.theta.shape[0]))
    thetas[0] = first_dtimestep.experience.theta

    for n_episode in xrange(1, n_episodes):
        thetas[n_episode] = next(cnts_dtimesteps)[1].experience.theta

    return thetas


def plot_2D_V(state_ranges, act_space, feature_vec, theta):
    pos_range = np.linspace(state_ranges[0][0], state_ranges[1][0], 40)
    vel_range = np.linspace(state_ranges[0][1], state_ranges[1][1], 40)
    act_range = np.arange(act_space.n)

    poss, vels, acts = np.meshgrid(pos_range, vel_range, act_range)

    ustate_ranges = np.frompyfunc(
                        lambda p, v, a: \
                            feature_vec(np.array([p, v]), a).dot(theta),
                        3, 1)

    Qs = ustate_ranges(poss, vels, acts)
    Vs = np.max(Qs, axis=2)

    print poss[:,:,0].shape, vels[:,:,0].shape, Vs.shape

    figure  = pyplot.figure(1)
    axes    = figure.gca(projection='3d')
    axes.plot_surface(poss[:,:,0], vels[:,:,0], Vs, rstride=1,
                        cstride=1, cmap=cm.Greys,
                        antialiased=False, linewidth=0)
    pyplot.show()

#def train(n_episodes, env, think, init_experience):
#    dtimestep = DTimestep(env.reset(), 0, init_experience, None)
#    for _ in xrange(n_episodes):
#        dtimestep = train_and_prep(dtimestep)
#
#    return V_from_Q(dtimestep.experience.Q)
