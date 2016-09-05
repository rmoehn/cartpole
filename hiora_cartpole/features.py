# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np

import gym
from tilecoding import representation

def make_feature(ndivs, ntilings):
    """

    Arguments:
        ntilings - int, number of tilings
    """
    # For each dimension, one row of offsets per tiling.
    unscaled_offsets = np.random.random_sample(4 * ntilings)\
                           .reshape((4, ntilings))
    # unscaled_offsets are from interval [0, 1[. Make them smaller than the
    # width of the divisions in every dimension.
    offsets = -1.0/np.array(ndivs) * unscaled_offsets

    cartpole = gym.make('CartPole-v0')

    state_tc = representation.TileCoding(
                   input_indices=[[0, 1, 2, 3]],
                   ntiles=ndivs,
                   ntilings=[ntilings],
                   state_range=np.array([cartpole.observation_space.low,
                                         cartpole.observation_space.high]),
                   offsets=np.array([offsets]),
                   hashing=None,
                   bias_term=False)

    action_vec = [np.array([-1]), np.array([0])] # This appears to be fastest.

    state_action_tc = representation.ConcatStateAction(state_tc)
    state_action_to_dense = representation.IndexToDense(state_action_tc)


    def feature_inner(state, action):
        return state_action_tc(state, action_vec[action])

    return feature_inner
