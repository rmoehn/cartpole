# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools

import numpy as np

class FourierFeatureVec(object):
    def __init__(self, action, feature_vec):
        self.action      = action
        self.feature_vec = feature_vec
        self.slice       = slice(action * feature_vec.shape[0],
                                 (action+1) * feature_vec.shape[0])


    def dot(self, vec):
        return np.dot(vec[self.slice], self.feature_vec)


    def add_to(self, vec):
        """

        Warning: Modifies vec.
        """
        vec[self.slice] += self.feature_vec


    def alphabounds_diffdot(self, prev, elig):
        """

        Credits: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        """
        return np.dot(elig[self.slice], self.feature_vec) \
                   - np.dot(elig[prev.slice], prev.feature_vec)



def make_feature_vec(state_ranges, n_acts, order):
    """

    Arguments:
        state_ranges – (2, n_dims) minima and maxima of possible state values
        n_acts – int, number of actions that can happen
        order – int, order of the Fourier basis

    Credits:
     - http://psthomas.com/papers/Konidaris2011a.pdf
     - https://github.com/amarack/python-rl/blob/master/pyrl/basis/fourier.py
    """
    n_dims    = state_ranges.shape[1]
    n_entries = (order + 1)**n_dims
    intervals = np.diff(state_ranges, axis=0)

    # All entries from cartesian product {0, …, order+1}^n_dims.
    c_matrix = np.array(
                   list( itertools.product(range(order+1), repeat=n_dims) ),
                   dtype=np.int32)

    assert n_entries == c_matrix.shape[0] # Sanity check.

    def feature_vec_dot_inner(state, action):
        """

        Arguments:
            action - int in {0, …, number of possible actions}
        """

        # Note: With the default C/row-major format it should be faster to put
        # the c₁, c₂, … in the rows of the matrix.

        # Bring all state input into the range [0, 1], the input range of the
        # Fourier basis functions.
        normalized_state = (state - state_ranges[0]) / intervals

        # Dot products of the feature vector with every c. → shape (n_entries,)
        dot_prods = np.dot(c_matrix, normalized_state.transpose())[:,0]

        # Apply Fourier basis functions.
        feature_v = np.cos(np.pi * dot_prods)

        # Sum up results, weighted, to give Fourier val.
        return FourierFeatureVec(action, feature_v)

    return n_acts * n_entries, feature_vec_dot_inner
