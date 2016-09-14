# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
from scipy import sparse

import gym
from tilecoding import representation

class TileCodingFeatureVec(object):
    def __init__(self, feature_vec):
        self.feature_vec = feature_vec


    def dot(self, vec):
        return self.feature_vec.dot(vec)[0]


    def add_to(self, vec):
        vec[self.feature_vec.indices] += 1.0


def make_feature_vec(ndivs, ntilings):
    """

    Arguments:
        ndivs    – (n_dim), number of divisions (in each tiling) for each of the
                   four input dimensions
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
                   ntiles=ndivs + [2],
                   ntilings=[ntilings],
                   state_range=np.array([cartpole.observation_space.low,
                                         cartpole.observation_space.high]),
                   offsets=np.array([offsets]),
                   hashing=None,
                   bias_term=False)

    state_inds_to_sparse = representation.IndexToBinarySparse(state_tc)


    def feature_vec_inner(state, action):
        state_fv = state_inds_to_sparse(state) # fv … feature vector
        # Note: Normally it would look like this:
        #
        #  tiling   0       1       2
        #  action = 0   1   0   1   0   1
        #           |---|---|---|---|---|---
        #
        # Here it looks like this:
        #
        # tiling    0   1   2   0   1   2
        # action =  0           1
        #           |---|---|---|---|---|---
        #
        # For the scalar product and the weights, this shouldn't make a
        # difference, though.
        if action:
            return TileCodingFeatureVec(
                       sparse.hstack(
                           [state_fv, sparse.csr_matrix(state_fv.shape)],
                           format='csr'))
        else:
            return TileCodingFeatureVec(
                       sparse.hstack(
                           [sparse.csr_matrix(state_fv.shape), state_fv],
                           format='csr'))

    return feature_vec_inner
