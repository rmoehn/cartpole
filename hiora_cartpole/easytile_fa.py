# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import operator

import numpy as np

import easytile


class EasyTileFeatureVec(object):
    def __init__(self, indices):
        self.indices = indices

    def dot(self, vec):
        return np.sum(vec[self.indices])


    def add_to(self, vec):
        vec[self.indices] += 1


    def alphabounds_diffdot(self, prev, elig):
        return np.sum(elig[self.indices] - elig[prev.indices])



def make_feature_vec(dim_ranges, n_acts, n_divss, n_tilings):
    n_divss          = np.array(n_divss)
    n_tilings        = np.array(n_tilings)
    n_dim            = dim_ranges.shape[1]
    unscaled_offsets = np.random.rand(n_tilings, n_dim)
    # unscaled_offsets are from interval [0, 1[. Make them smaller than the
    # width of the divisions in every dimension.
    div_widths = np.diff(dim_ranges, axis=0) / (n_divss - 1)
    offsetss   = -div_widths * unscaled_offsets
    feature_fn = easytile.make_feature_fn(dim_ranges, n_tilings, n_divss,
                                          offsetss)

    n_weights_per_act = n_tilings * reduce(operator.mul, n_divss)

    def feature_vec_inner(state, action):
        indices = feature_fn(state)
        return EasyTileFeatureVec(indices + action * n_weights_per_act)

    return n_weights_per_act * n_acts, feature_vec_inner
