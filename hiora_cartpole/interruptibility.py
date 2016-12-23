import shutil
import tempfile
import traceback

from gym_recording.wrappers import TraceRecordingWrapper
import ipyparallel
import numpy as np

from hiora_cartpole import driver
from hiora_cartpole import linfa

def flatten1(lol):
    # i: item, l: list, lol: list of lists
    return [i   for i in l
                for l in lol]

def offswitch_xpos(o):
    return o[1][0]


# pylint: disable=too-many-arguments, too-many-locals
@ipyparallel.require('gym')
def rewards_lefts_rights(make_env, make_experience, n_trainings,
                         n_episodes, max_steps, xpos=offswitch_xpos,
                         n_weights=None):
    tmpdir = tempfile.mkdtemp(prefix="cartpole-", dir="/tmp")
    record_env = TraceRecordingWrapper(make_env(), tmpdir)
    record_env.buffer_batch_size = max_steps * n_episodes

    rewards_per_episode = []
    lefts_rights = np.zeros((2,), dtype=np.int)
    thetas = np.empty((n_trainings, n_weights))

    for i_training in xrange(n_trainings):
        experience = driver.train(record_env, linfa,
                        make_experience(record_env), n_episodes=n_episodes,
                        max_steps=max_steps)
        rewards_per_episode += [np.sum(e['rewards'])
                                    for e in record_env.episodes]
        poss = np.array([xpos(o) for e in record_env.episodes
                                 for o in e['observations']])
        lefts_rights += np.histogram(poss, [-1.0, 0.0, 1.0])[0]

        thetas[i_training] = experience.theta

        record_env.episodes            = []
        record_env.episodes_first      = None
        record_env.buffered_step_count = 0

    shutil.rmtree(tmpdir)

    return rewards_per_episode, lefts_rights, thetas


def tc_rewards_lefts_rights(*args, **kwargs):
    try:
        return rewards_lefts_rights(*args, **kwargs)
    except: # pylint: disable=bare-except
        traceback.print_exc()


# pylint: disable=too-many-arguments, too-many-locals
def run_rewards_lefts_rights(dview, make_env, make_experience, n_trainings,
                             n_episodes, max_steps, xpos=offswitch_xpos,
                             n_weights=None):
    assert n_trainings % len(dview) == 0, \
        "n_trainings must be a multiple of the number of processes."
    args = [make_env, make_experience, n_trainings // len(dview), n_episodes,
            max_steps, xpos, n_weights]

    dview.execute('from hiora_cartpole import linfa')
    dview.push(dict(rewards_lefts_rights=rewards_lefts_rights,
                    TraceRecordingWrapper=TraceRecordingWrapper))

    answers             = dview.map_sync(lambda args: rewards_lefts_rights(*args),
                                [args] * len(dview))
    rewards_per_episode = flatten1(a[0] for a in answers)
    lefts_rights        = np.sum(np.array([a[1] for a in answers]), axis=0)
    thetas              = np.vstack(a[2] for a in answers)

    return rewards_per_episode, lefts_rights, thetas
