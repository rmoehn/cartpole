import argparse
import functools
import logging

logging.basicConfig(level=logging.DEBUG)

import gym
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot
import numpy as np

import gym_ext.tools as gym_tools
from hiora_cartpole import driver
from hiora_cartpole import fourier_fa
#from hiora_cartpole import easytile_fa
from hiora_cartpole import linfa
from hiora_cartpole import offswitch_hfa

#### Arg parsing

parser = argparse.ArgumentParser()
parser.add_argument('--is-monitored', action='store_true')
parser.add_argument('obs_path')
parser.add_argument('act_path')
args = parser.parse_args()

#### The real stuff

clipped_high = np.array([2.5, 4.4, 0.28, 3.9])
clipped_low  = -clipped_high
state_ranges = np.array([clipped_low, clipped_high])

env0 = gym.make('CartPole-v0')
#env0.seed(42)

four_n_weights, four_feature_vec \
    = fourier_fa.make_feature_vec(state_ranges,
                                  n_acts=2,
                                  order=3)
#four_n_weights, four_feature_vec \
#    = easytile_fa.make_feature_vec(np.array([env.low, env.high]), 3, [9, 9], 5)

ofour_n_weights, ofour_feature_vec \
    = offswitch_hfa.make_feature_vec(four_feature_vec, four_n_weights)

skip_offswitch_clip = functools.partial(
                          gym_tools.apply_to_snd,
                          functools.partial(gym_tools.warning_clip_obs, ranges=state_ranges))

experience0 = linfa.init(lmbda=0.9,
                         init_alpha=0.001,
                         epsi=0.1,
                         feature_vec=four_feature_vec,
                         n_weights=four_n_weights,
                         act_space=env0.action_space,
                         theta=None,
                         is_use_alpha_bounds=True,
                         map_obs=functools.partial(gym_tools.warning_clip_obs, ranges=state_ranges),
                         choose_action=linfa.choose_action_Sarsa)

n_episodes = 200

#np.random.seed(42)

if args.is_monitored:
    env0.monitor.start("/tmp/cartpole-experiment-1", force=True)

experience0, steps_per_episode0, alpha_per_episode0, observations0, actions0 \
    = driver.train(env0, linfa, experience0, n_episodes=n_episodes,
            max_steps=200, is_render=False, is_wrapup_at_max_steps=False)

if args.is_monitored:
    env0.monitor.close()

np.save(args.obs_path, observations0)
np.save(args.act_path, actions0)

fig = pyplot.figure(figsize=(5,8))
ax01 = fig.add_subplot(211)
ax01.plot(steps_per_episode0, color='b')
ax02 = ax01.twinx()
ax02.plot(alpha_per_episode0, color='r')
ax03 = fig.add_subplot(212)
ax03.plot(experience0.theta)
pyplot.show()
