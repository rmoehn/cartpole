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

# pylint: disable=redefined-outer-name
def train(env, experience, n_episodes, max_steps, is_render=False):
    steps_per_episode = np.zeros(n_episodes, dtype=np.int32)

    for n_episode in xrange(n_episodes):
        observation = env.reset()
        reward      = 0
        done        = False

        for t in xrange(max_steps):
            is_render and env.render() # pylint: disable=expression-not-assigned
            experience, action = linfa.think(experience, observation, reward,
                                             done)
            observation, reward, done, _ = env.step(action)

            if done:
                steps_per_episode[n_episode] = t
                experience = linfa.wrapup(experience, observation, reward)
                break

    return experience, steps_per_episode

#hard_earned_theta = np.copy(experience.theta)
#np.savez_compressed("hard-earned-theta", hard_earned_theta)
