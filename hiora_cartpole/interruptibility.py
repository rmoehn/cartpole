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


def count_lefts_rights(xss):
    xs_upto_cross = itertools.chain(
                        *[itertools.takewhile(lambda x: x <= 1.0, xs)
                             for xs in xss])
    return np.histogram(np.fromiter(xs_upto_cross, np.float64),
                        [-1.0, 0.0, 1.0])[0]
