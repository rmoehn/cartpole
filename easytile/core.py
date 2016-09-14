# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import functools
from unittest import TestCase

import numpy as np

def check_args(dim_ranges, n_tilings, n_divss, offsetss):
    tc = TestCase('__init__')

    n_dim = dim_ranges.shape[0]
    tc.assertEqual(2,                  dim_ranges.shape[1])
    tc.assertEqual(tuple([n_dim]),     n_divss.shape)
    tc.assertEqual((n_tilings, n_dim), offsetss.shape)

    div_widths = np.diff(dim_ranges, axis=1) / (n_divss - 1)

    assert all(-div_widths < offsetss), "Offsets must be less than division widths."
    assert all(offsetss <= 0), "Offsets must be <= 0."


def raw_feature_fn(dim_ranges, n_tilings, n_divss, offsetss, state):
    """

    Note with example::

                      0    1    2    3    4
        Tiling 1:     |    |    |    |    |    |
        Tiling 2:   |    |    |    |    |    |
        Input space:  ↑                   ↑
                      0                   20

    In this case we have::

        dim_ranges = [[0, 20]] (call them min and max for now)
        n_tilings  = 2
        n_divss    = 5
        offsetss   = [[0], [-2]]

    Each division is 5 = (max - min) / (n_divs - 1) broad. This might surprise
    – after all we have 5 divisions, why not divide the input space by 5 and get
    divisions of width 4? But this works only for offset 0 and only if we had
    exclusive max values. For other offsets and for the inclusive max value that
    we have, we always need one extra division. Note that the strict inequality
    (div_width < offset) ensures that we don't need even one extra division for
    offset grids.


    Arguments:
        dim_ranges – shape: (n_dim, 2) (min and max value for each dimension)
        n_tilings  – int > 0
        n_divss    – shape: (n_dim)
                     (number of divisions for each dimension (same for all
                     tilings), for each dimension: div_width = (max - min) /
                     (n_divs - 1))
        offsetss   – shape: (n_tilings, n_dim)
                     (offset in each dimension for each tiling, for each
                     dimension and tiling: div_width < offset <= 0)
        state      – shape: (n_dim)

    Returns:
        shape: (n_tilings * product(n_divss))
    """

    # TODO: Centered is misleading. We left-align the values at zero, but don't
    # center them at zero.
    offset     = state - offsetss           # offset state for each tiling
    centered   = offset - dim_ranges[:, 0]  # subtract min values
    normalized = centered / np.diff(dim_ranges, axis=1)
                                            # divide by space width
    scaled     = normalized * (n_divss - 1) # distribute over divisions
    div_coords = scaled.astype(np.int)      # floor to get division coords
    indices    = np.ravel_multi_index( div_coords.transpose(), n_divss )
        # transform multi-dimensional coordinates to scalar tile indices

    res = np.zeros(n_tilings * np.product(n_divss), dtype=np.bool)
    res[ np.array(indices) ] = 1
        # Turn on features indicated by indices.
    return res


def make_feature_fn(dim_ranges, n_tilings, n_divss, offsetss):
    args = (np.array(dim_ranges), n_tilings, np.array(n_divss),
            np.array(offsetss))

    check_args(*args)

    return functools.partial(raw_feature_fn, *args)
