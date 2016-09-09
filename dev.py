# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np

from hiora_cartpole import features

import gym




feature_vec = features.make_feature_vec([10, 10, 10, 10], 4)


cartpole = gym.make('CartPole-v0')

#fv = feature_vec(cartpole.observation_space.sample(), cartpole.action_space.sample())

from hiora_cartpole import linfa
experience = linfa.init(lmbda=0.3,
                        alpha=0.01,
                        epsi=0.05,
                        feature_vec=feature_vec,
                        n_weights=80000,
                        act_space=cartpole.action_space)

env = cartpole

for n_episode in range(10000):
    observation = env.reset()
    reward      = 0
    done        = False

    for t in range(100):
        experience, action = linfa.think(experience, observation, reward, done)
        observation, reward, done, info = env.step(action)

        if done:
            print "Episode finished after {} timesteps".format(t+1)
            experience = linfa.wrapup(experience, observation, reward)
            break

hard_earned_theta = np.copy(experience.theta)

np.save("hard-earned-thetauuu", hard_earned_theta, allow_pickle=False)
