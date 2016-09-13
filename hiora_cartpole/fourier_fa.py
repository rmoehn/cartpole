# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools

import numpy as np

def make_feature_vec(state_ranges, order):
    """

    Arguments:
        state_ranges – (2, n_dims) minima and maxima of possible state values
        n_acts – int, number of actions that can happen
        order – int, order of the Fourier basis

    Credits:
     - http://psthomas.com/papers/Konidaris2011a.pdf
     - https://github.com/amarack/python-rl/blob/master/pyrl/basis/fourier.py
    """
    n_dims    = state_ranges[1]
    n_entries = (n_dims + 1)**order

    # All entries from cartesian product {0, …, order+1}^n_dims.
    c_matrix = np.array(
                   list( itertools.product(range(order+1), repeat=n_dims) ),
                   dtype=np.int32)

    def feature_vec_dot_inner(state, action, weights):
        """

        Arguments:
            action - int in {0, …, number of possible actions}
        """

        # Note: With the default C/row-major format it should be faster to put
        # the c₁, c₂, … in the rows of the matrix.

        # Bring all state input into the range [0, 1], the input range of the
        # Fourier basis functions.
        normalized_state = (state - state_ranges[0]) \
                               / np.diff(state_ranges, axis=1)

        # Dot products of the feature vector with every c. → shape (n_entries,)
        dot_prods = np.dot(c_matrix, normalized_state.transpose())[:,0]

        # Apply Fourier basis functions.
        feature_v = np.cos(np.pi * dot_prods)

        # Sum up results, weighted, to give Fourier val.
        return np.dot(weights[action * n_entries:(action + 1) * n_entries - 1],\
                      feature_v)

    return feature_vec_dot_inner
