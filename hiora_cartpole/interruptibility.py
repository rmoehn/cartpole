import itertools
import traceback

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper
import numpy as np

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


# Note: For some reason the number of x values per episode is (steps for that
# episode + 2)
def split_per_episode(steps_per_episode, xs):
    idxs = np.cumsum(steps_per_episode + 2)[:-1]
    return np.split(xs, idxs)


def remove_xs_after_crosses(steps, xss):
    xss_per_episode = (xs_this_episode
                            for spe, xs in zip(steps, xss)
                            for xs_this_episode in split_per_episode(spe, xs))
    xs_upto_cross = itertools.chain(
                        *[itertools.takewhile(lambda x: x <= 1.0, xs)
                             for xs in xss_per_episode])
    return np.fromiter(xs_upto_cross, np.float64)
