import multiprocessing
import shutil
import tempfile
import traceback

import gym.wrappers
import numpy as np

from hiora_cartpole import driver
from hiora_cartpole import linfa

def rewards_lefts_rights(make_env, make_experience, n_trainings,
                         n_episodes, max_steps):
    tmpdir = tempfile.mkdtemp(prefix="cartpole-", dir="/tmp")
    record_env = gym.wrappers.TraceRecordingWrapper(make_env(), tmpdir)
    record_env.buffer_batch_size = max_steps * n_episodes

    rewards_per_episode = []
    lefts_rights = np.zeros((2,), dtype=np.int)

    for _ in xrange(n_trainings):
        driver.train(record_env, linfa, make_experience(),
                     n_episodes=n_episodes, max_steps=max_steps)
        rewards_per_episode += [np.sum(e['rewards'])
                                    for e in record_env.episodes]
        poss = np.array([o[1][0] for e in record_env.episodes
                                 for o in e['observations']])
        lefts_rights += np.histogram(poss, [-1.0, 0.0, 1.0])[0]

        record_env.episodes            = []
        record_env.episodes_first      = None
        record_env.buffered_step_count = 0

    shutil.rmtree(tmpdir)

    return rewards_per_episode, lefts_rights


def tc_rewards_lefts_rights(*args, **kwargs):
    try:
        return rewards_lefts_rights(*args, **kwargs)
    except: # pylint: disable=bare-except
        traceback.print_exc()


# pylint: disable=too-many-arguments
def run_rewards_lefts_rights(make_env, make_experience, n_procs, n_trainings,
                             n_episodes, max_steps):
    pool = multiprocessing.Pool(n_procs)
    args = [make_env, make_experience, n_trainings // n_procs, n_episodes,
            max_steps]

    results             = [pool.apply_async(tc_rewards_lefts_rights, args)
                               for _ in xrange(n_procs)]
    answers             = [r.get() for r in results]
    rewards_per_episode = [r for a in answers for r in a[0]]
    lefts_rights        = np.sum(np.array([a[1] for a in answers]), axis=0)

    return rewards_per_episode, lefts_rights
