# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import functools
from unittest import TestCase

import numpy as np

def check_args(dim_ranges, dim_widths, n_tilings, n_divss, offsetss):
    tc = TestCase('__init__')

    n_dim = dim_ranges.shape[1]
    tc.assertEqual(2,                  dim_ranges.shape[0])
    tc.assertEqual(tuple([n_dim]),     n_divss.shape)
    tc.assertEqual((n_tilings, n_dim), offsetss.shape)

    div_widths = dim_widths / (n_divss - 1)

    assert np.all(-offsetss < div_widths), \
           "Offsets must be less than division widths."
    assert np.all(offsetss <= 0), "Offsets must be <= 0."




def make_feature_fn(dim_ranges, n_tilings, n_divss, offsetss):
    dim_widths = np.diff(dim_ranges, axis=0)

    check_args(dim_ranges, dim_widths, n_tilings, n_divss, offsetss)

    n_tiles_per_tiling = np.product(n_divss)
    n_total_tiles      = n_tiles_per_tiling * n_tilings

    def feature_fn_inner(state):
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
            dim_ranges – shape: (2, n_dim) (min and max value for each dimension)
            n_tilings  – == offsetss.shape[0]
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
        x  = state - offsetss # offset state for each tiling
        x -= dim_ranges[0]    # left-align at zero (or more, because of offset)
        x /= dim_widths       # divide by space width to normalize (range 1)
        x *= (n_divss - 1)    # distribute over divisions
        div_coords = x.astype(np.int) # floor to get division coords
        indices    = np.ravel_multi_index( div_coords.transpose(), n_divss )
            # transform multi-dimensional coordinates to scalar tile indices
        indices += np.arange(0, n_total_tiles, n_tiles_per_tiling)
            # distribute tile indices over the output sections of their tilings

        return indices

    return feature_fn_inner
