# -*- coding: utf-8 -*-

from __future__ import unicode_literals

# HO … higher-order
class SliceHOFeatureVec(object):
    def __init__(self, slice_i, entries_per_slice, feature_vec):
        self.feature_vec = feature_vec
        self.slice       = slice(slice_i * entries_per_slice,
                                 (slice_i+1) * entries_per_slice)


    def dot(self, vec):
        return self.feature_vec.dot(vec[self.slice])


    def add_to(self, vec):
        """

        Warning: Modifies vec.
        """
        self.feature_vec.add_to(vec[self.slice])


    def alphabounds_diffdot(self, prev, elig):
        """

        Credits: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        """
        return self.dot(elig) - prev.dot(elig)


def make_feature_vec(feature_vec, n_weights):
    def feature_vec_inner(state, action):
        return SliceHOFeatureVec(state[0], n_weights,
                                 feature_vec(state[1], action))

    return n_weights * 2, feature_vec_inner
