# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from unittest import TestCase

import numpy as np

import easytile.core as mut

class NumpyTestCase(TestCase):
    def assertArrayEqual(self, a, b): # pylint: disable=no-self-use
        if not np.array_equal(a, b):
            raise AssertionError(
                "Arrays differ: {} != {}\nDifference: {}".format(a, b, b - a))


class TestFeature(NumpyTestCase):
    def setUp(self):
        self.feature_1dim = mut.make_feature_fn(
                                [[0, 20]], 3, [5], [[0], [-2], [-4.5]])

    def test_features_1dim(self):
        self.assertArrayEqual(
                self.feature_1dim(11),
                np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                         dtype=np.bool))


class TestMakeFeatureFn(TestCase):
    def test_bad_offset_fails_1dim(self):
        with self.assertRaises(AssertionError):
            mut.make_feature_fn([[-5, 34]], 3, [5], [[-10]])
