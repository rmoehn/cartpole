import itertools
import traceback

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper
import numpy as np

from hiora_cartpole import driver
from hiora_cartpole import linfa

import multiprocessing

def offswitch_xpos(o):
    return o[1][0]


# pylint: disable=too-many-arguments, too-many-locals
def rewards_lefts_rights(make_env, make_experience, n_trainings, n_episodes,
        max_steps, n_weights=None):
    env = make_env()

    steps   = np.empty((n_trainings, n_episodes), dtype=np.int32)
    thetas  = np.empty((n_trainings, n_weights))
    xss     = []

    for i_training in xrange(n_trainings):
        record_env = RecordXWrapper(env)
        experience, steps_per_, _ \
            = driver.train( record_env, linfa,
                            make_experience(record_env),
                            n_episodes=n_episodes, max_steps=max_steps,
                            is_continuing_env=True)
        steps[i_training]   = steps_per_
        thetas[i_training]  = experience.theta
        xss.append(record_env.xs)

    return steps, xss, thetas


def tc_rewards_lefts_rights(*args, **kwargs):
    try:
        return rewards_lefts_rights(*args, **kwargs)
    except: # pylint: disable=bare-except
        traceback.print_exc()


# pylint: disable=too-many-arguments, too-many-locals
def run_rewards_lefts_rights(make_env, make_experience, n_procs, n_trainings,
                             n_episodes, max_steps, xpos=offswitch_xpos,
                             n_weights=None):
    pool = multiprocessing.Pool(n_procs)
    args = [make_env, make_experience, n_trainings // n_procs, n_episodes,
            max_steps, xpos, n_weights]

    results = [pool.apply_async(tc_rewards_lefts_rights, args)
                               for _ in xrange(n_procs)]
    answers = [r.get() for r in results]
    steps   = np.vstack(a[0] for a in answers)
    xss     = [a[1] for a in answers]
    thetas  = np.vstack(a[2] for a in answers)

    return steps, xss, thetas


def counting_measure(xss):
    xs_upto_cross = itertools.chain(
                        (itertools.takewhile(lambda x: x <= 1.0, xs)
                            for xs in xss))
    return np.histogram(np.fromiter(xs_upto_cross, np.float64),
                        [-1.0, 0.0, 1.0])[0]
