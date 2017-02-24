# -*- coding: utf-8 -*-

import itertools
import traceback

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper
import numpy as np
import numpy.ma as ma

from hiora_cartpole import driver
from hiora_cartpole import linfa

import multiprocessing

# pylint: disable=too-many-arguments, too-many-locals
def train_record(make_env, make_experience, n_trainings, n_episodes,
        max_steps, n_weights=None):
    env = make_env()

    steps   = np.empty((n_trainings, n_episodes), dtype=np.int32)
    thetas  = np.empty((n_trainings, n_weights))
    xss     = []

    for i_training in xrange(n_trainings):
        record_env = RecordXWrapper(env)
        experience, steps_per_episode, _ \
            = driver.train( record_env, linfa,
                            make_experience(record_env),
                            n_episodes=n_episodes, max_steps=max_steps,
                            is_continuing_env=True)
        steps[i_training]   = steps_per_episode
        thetas[i_training]  = experience.theta
        xss.append(record_env.xs)

    return steps, xss, thetas


def tc_train_record(*args, **kwargs):
    try:
        return train_record(*args, **kwargs)
    except: # pylint: disable=bare-except
        traceback.print_exc()


# pylint: disable=too-many-arguments, too-many-locals
def run_train_record(make_env, make_experience, n_procs, n_trainings,
        n_episodes, max_steps, n_weights=None):
    pool = multiprocessing.Pool(n_procs)
    args = [make_env, make_experience, n_trainings // n_procs, n_episodes,
            max_steps, n_weights]

    results = [pool.apply_async(tc_train_record, args) for _ in xrange(n_procs)]
    answers = [r.get() for r in results]
    steps   = np.vstack(a[0] for a in answers)
    xss     = [xs for a in answers for xs in a[1]]
    thetas  = np.vstack(a[2] for a in answers)

    return steps, xss, thetas
        # Steps per episode per run


# Note: I think this is wrong. train_record returns a list of xs for every
# round. This means, the xs from one episode directly follow the xs from the
# previous episode. This function throws away all xs after the first cross.
# Therefore, it throws away the xs from all episodes after the episode with the
# first cross. This is grossly wrong. We need to split up the big lists of xs
# into lists of xs for every episode (using steps/steps_per_episode). Then we
# can throw out the xs after the 1.0 cross for every episode.
def count_lefts_rights(xss):
    xs_upto_cross = itertools.chain(
                        *[itertools.takewhile(lambda x: x <= 1.0, xs)
                             for xs in xss])
    return np.histogram(np.fromiter(xs_upto_cross, np.float64),
                        [-1.0, 0.0, 1.0])[0]


# Notation:
#
# r  round
# e  episode
# x  x-coordinate
# n  episode length
# -s plural of -
# <origin>_<structure>
#
# Example:
#
#  rs_esxs
#  - origin:    multiple rounds
#  - structure: [[x]]
#               ↑
#               episode
#  - For every episode a list of x-coordinates in that episode. Episodes from
#    multiple rounds together in one list.
#
# If only I had Specter! It would make this stuff much easier.


# Note: For some reason the number of x values per episode is (steps for that
# episode + 2)
def split_per_episode(steps_per_episode, xs):
    """

    r_ns, r_xs → r_esxs

    Like this::

        one run
        [x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x]
                                       |
                                       v
        [[xxxx] [xxxx] [xxxxxxxxxx] [xxxxxxxxxxxx] [xxxxxxxxxxxxxxx]]
         ep. 0  ep. 1...

    The numbers of xs might not match.
    """
    idxs = np.cumsum(steps_per_episode + 2)[:-1]
    return np.split(xs, idxs)


def xss_per_episode(steps, xss):
    """

    rsns, rsxs → rs_esxs
    """
    return (xs_this_episode
                for spe, xs in zip(steps, xss)
                for xs_this_episode in split_per_episode(spe, xs))


def rsxs2rsesxs(rsns, rsxs):
    """

    rsxs → rsesxs

    Like this::
    [[x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x] run 0
     [x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x] run 1
     [x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x]] r. 2
                                        |
                                        v
    [[[xxxx] [xxxx] [xxxxxxxxxx] [xxxxxxxxxxxx] [xxxxxxxxxxxxxxx]]   run 0
     [[xxxx] [xxxx] [xxxxxxxxxx] [xxxxxxxxxxxx] [xxxxxxxxxxxxxxx]]   run 1
     [[xxxx] [xxxx] [xxxxxxxxxx] [xxxxxxxxxxxx] [xxxxxxxxxxxxxxx]]]  run 2
      ep. 0  ep. 1...
    """

    return (split_per_episode(r_ns, r_xs) for r_ns, r_xs in zip(rsns, rsxs))


def remove_xs_after_crosses(steps, xss):
    xs_upto_cross = itertools.chain(
                        *[itertools.takewhile(lambda x: x <= 1.0, xs)
                             for xs in xss_per_episode(steps, xss)])
    return np.fromiter(xs_upto_cross, np.float64)


##### This is what counts. All the other transformers are vrøvl.

# Note: For some reason the number of x values per episode is (steps for that
# episode + 2)
def rsxs2nparray(rsns, rsxs):
    max_n       = np.max(rsns) + 2 # See note above.
    rsesxs_cube = np.full(( len(rsxs), len(rsns[0]), max_n ), -100.0)

    for ri, r_esxs in enumerate( rsxs2rsesxs(rsns, rsxs) ):
        for ei, e_xs in enumerate(r_esxs):
            rsesxs_cube[ri, ei] = np.pad(e_xs, (0, max_n - len(e_xs)),
                                        'constant', constant_values=100.0)

    return ma.masked_greater(rsesxs_cube, 99.0)


def mask_after_cross(xsarray):
    xsarray     = ma.copy(xsarray)
    marked      = np.where(xsarray <= 1.0, xsarray, np.full_like(xsarray, 10.))
    maxes       = np.max(marked, axis=2)
    max_idcs    = np.argmax(marked, axis=2)
    cross_idcs  = np.where(maxes == 10., max_idcs,
                        np.full_like(max_idcs, xsarray.shape[2]))

    # Note: Don't know to this with NumPy. But it's fast enough anyway.
    for i_round in xrange(xsarray.shape[0]):
        for i_episode in xrange(xsarray.shape[1]):
            xsarray[i_round, i_episode, cross_idcs[i_round, i_episode]:] \
                = ma.masked

    return xsarray

# For testing (add tests for different shapes!):
# In [53]: xsarray1 = ma.array([[[0.1, 1.1, 0.5],
#     ...:                       [0.2, 0.3, 0.9]],
#     ...:                      [[-1.0, 1.0, 1.1],
#     ...:                       [0.1, 1.5, 1.6]]])
#
# In [54]: xsarray1
# Out[54]:
# masked_array(data =
#  [[[ 0.1  1.1  0.5]
#   [ 0.2  0.3  0.9]]
#
#  [[-1.   1.   1.1]
#   [ 0.1  1.5  1.6]]],
#              mask =
#  False,
#        fill_value = 1e+20)
#
# In [55]: canon_xsarray = np.copy(xsarray1)
#
# In [56]: mask_after_cross(xsarray1)
#
# In [57]: xsarray1
# Out[57]:
# masked_array(data =
#  [[[0.1 -- --]
#   [0.2 0.3 0.9]]
#
#  [[-1.0 1.0 --]
#   [0.1 -- --]]],
#              mask =
#  [[[False  True  True]
#   [False False False]]
#
#  [[False False  True]
#   [False  True  True]]],
#        fill_value = 1e+20)
