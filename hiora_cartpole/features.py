# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import operator

import numpy as np
from scipy import sparse

from tilecoding import representation

class TileCodingFeatureVec(object):
    def __init__(self, feature_vec):
        self.feature_vec = feature_vec


    def dot(self, vec):
        return self.feature_vec.dot(vec)[0]


    def add_to(self, vec):
        vec[self.feature_vec.indices] += 1.0


    def alphabounds_diffdot(self, prev, elig):
        return self.dot(elig) - prev.dot(elig)


def make_feature_vec(state_ranges, n_acts, ndivs, ntilings):
    """

    Arguments:
        ndivs    – (n_dim), number of divisions (in each tiling) for each of the
                   four input dimensions
        ntilings - int, number of tilings
    """
    n_dim = state_ranges.shape[1]
    # For each dimension, one row of offsets per tiling.
    # Note: if you use np.random.rand, you don't need the reshape.
    unscaled_offsets = np.random.random_sample(n_dim * ntilings)\
                           .reshape((n_dim, ntilings))
    # unscaled_offsets are from interval [0, 1[. Make them smaller than the
    # width of the divisions in every dimension.
    offsets = -1.0/np.array(ndivs)[None,:].transpose() * unscaled_offsets

    state_tc = representation.TileCoding(
                   input_indices=[range(n_dim)],
                   ntiles=ndivs,
                   ntilings=[ntilings],
                   state_range=state_ranges,
                   offsets=np.array([offsets]),
                   hashing=None,
                   bias_term=False)

    state_inds_to_sparse = representation.IndexToBinarySparse(state_tc)

    n_weights_per_act = ntilings * reduce(operator.mul, ndivs)
    n_weights         = n_acts * n_weights_per_act


    def feature_vec_inner(state, action):
        state_fv = state_inds_to_sparse(state) # fv … feature vector
        # Note: Normally it would look like this (n_acts == 2):
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

        n_after    = n_acts - action - 1
        n_before   = action
        return TileCodingFeatureVec(
                   sparse.hstack(
                       [sparse.csr_matrix((1, n_before * n_weights_per_act)),
                        state_fv,
                        sparse.csr_matrix((1, n_after * n_weights_per_act))],
                       format='csr'))

    return n_weights, feature_vec_inner
