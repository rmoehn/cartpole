import functools
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Bla")

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

clipped_high = np.array([2.5, 3.6, 0.28, 3.7])
clipped_low  = -clipped_high
state_ranges = np.array([clipped_low, clipped_high])

np.random.seed(42)
env0 = gym.make('CartPole-v0')
env0.seed(42)

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

n_episodes = 100

print np.random.rand(4)
experience0, steps_per_episode0, alpha_per_episode0, observations0, actions0 \
    = driver.train(env0, linfa, experience0, n_episodes=n_episodes, max_steps=200, is_render=False)

np.random.seed(42)
env1 = gym.make('CartPole-v0')
env1.seed(42)

experience1 = linfa.init(lmbda=0.9,
                         init_alpha=0.001,
                         epsi=0.1,
                         feature_vec=four_feature_vec,
                         n_weights=four_n_weights,
                         act_space=env1.action_space,
                         theta=None,
                         is_use_alpha_bounds=True,
                         map_obs=functools.partial(gym_tools.warning_clip_obs, ranges=state_ranges),
                         choose_action=linfa.choose_action_Sarsa)


#env.monitor.start("/tmp/cartpole-experiment-1", force=True)

print np.random.rand(4)
experience1, steps_per_episode1, alpha_per_episode1, observations1, actions1 \
    = driver.train(env1, linfa, experience1, n_episodes=n_episodes, max_steps=200, is_render=False)

#env.monitor.close()

for e in xrange(n_episodes):
    for t in xrange(200):
        is_break = False
        if actions0[e][t] != actions1[e][t]:
            print "%3d %3d: actions differ" % (e, t)
            is_break = True

        if not np.all(observations0[e][t] == observations1[e][t]):
            print "%3d %3d: %s != %s" % (e, t, observations0[e][t], observations1[e][t])
            is_break = True

        if is_break:
            break

fig = pyplot.figure(figsize=(10,8))
ax01 = fig.add_subplot(221)
ax01.plot(steps_per_episode0, color='b')
ax02 = ax01.twinx()
ax02.plot(alpha_per_episode0, color='r')
ax03 = fig.add_subplot(222)
ax03.plot(experience0.theta)
ax11 = fig.add_subplot(223)
ax11.plot(steps_per_episode1, color='b')
ax12 = ax11.twinx()
ax12.plot(alpha_per_episode1, color='r')
ax13 = fig.add_subplot(224)
ax13.plot(experience1.theta)
pyplot.show()
