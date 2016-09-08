# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools

import numpy as np
import pyrsistent

import easy21.core as easy21

LinfaExperience = pyrsistent.immutable(
                    'feature, theta, E, N0, epsi, alpha, lmbda, p_obs, p_act')


def init(lmbda, alpha, feature):
    """

    Arguments:
        feature - NPArray mapping dealer sums (axis 0), player sums (axis 1) and
                  actions (axis 2) to feature vectors
    """
    return LinfaExperience(feature=feature,
                           theta=np.zeros(feature.shape[3]),
                           E=np.zeros(feature.shape[3]),
                           N0=100,
                           epsi=0.05,
                           alpha=alpha,
                           lmbda=lmbda,
                           p_obs=None, # p … previous
                           p_act=None)


#### Make a feature lookup table

# Note: This is not what the exercise specifies, I think, but I don't understand
# how that what the exercise specifies makes sense. As I understand it, the
# features vectors in the the exercise are 36-element vectors in which exactly
# one element is 1 . The 1 indicates that the state is in certain dealer card
# intervals, certain player card intervals and that we've chose a certain
# action. Why not encode these separately? This is what I've done here. I'll see
# if it works or not.
#
# Note about note: I was wrong stating that exactly one element is 1. For
# example, dealer sum 4, player sum 6, action STICK turns on the ([1, 4], [1,
# 6], hit), ([1, 4], [4, 9], hit), ([4, 7], [1, 6], hit) and ([4, 7], [4, 9],
# hit) features.
def feature_slow(o, a):
    d = o.dealer_sum
    p = o.player_sum
    booleans = [1 <= d <= 4, 4 <= d <= 7, 7 <= d <= 10,
                1 <= p <= 6, 4 <= p <= 9, 7 <= p <= 12, 10 <= p <= 15,
                13 <= p <= 18, 16 <= p <= 21,
                a == easy21.Action.HIT] # Then 1/True indicates HIT.
    return np.array(booleans, np.bool)


def is_in_interval(n, iv):
    return iv[0] <= n <= iv[1]


# Note: This is what the exercise specifies. Actually it works much better than
# what I used. The encodings are equivalent, i.e. if you have a feature vector
# in their format, you can convert it into one in my format and vice versa. The
# difference is that their feature vector is longer, so that we can use more
# weights. Is this the only reason why it works much better?
def ex_feature_slow(o, a):
    dealer_intervals = [[1, 4], [4, 7], [7, 10]]
    player_intervals = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
    actions          = [easy21.Action.STICK, easy21.Action.HIT]
    d = o.dealer_sum
    p = o.player_sum
    return np.array([is_in_interval(d, div) and is_in_interval(p, piv)
                             and a == this_a
                         for (div, piv, this_a)
                         in itertools.product(dealer_intervals, player_intervals,
                                              actions)],
                    np.bool)


def prepare_feature(ao_to_feature):
    return np.array([
                        [
                            [ao_to_feature(easy21.Observation(d, p), a)
                                 for a in [easy21.Action.STICK,
                                           easy21.Action.HIT]
                            ]
                            for p in xrange(1, 22)
                        ]
                        for d in xrange(1, 11)
                    ])


def true_with_prob(p):
    return np.random.choice(2, p=[1-p, p])


def choose_action(e, o):
    if true_with_prob(e.epsi):
        return easy21.rand_action()
    else:
        stick_return = e.feature[o.dealer_sum - 1, o.player_sum - 1,
                                 easy21.Action.STICK].dot(e.theta)
        hit_return   = e.feature[o.dealer_sum - 1, o.player_sum - 1,
                                 easy21.Action.HIT].dot(e.theta)
        return easy21.Action.STICK if stick_return > hit_return \
                                   else easy21.Action.HIT
            # Python has not built-in readable argmax. numpy would be overkill.



def think(e, o, r, done=False):
    """

    Args:
        e … experience
        o … observation
        r … reward
    """

    if not done:
        a     = choose_action(e, o) # action
        feat  = e.feature[o.dealer_sum - 1, o.player_sum - 1, a]
        Qnext = feat.dot(e.theta)
            # expected Q of next action
    else:
        a     = None
        Qnext = 0

    if e.p_obs: # Except for first timestep.
        p_feat = e.feature[e.p_obs.dealer_sum - 1, e.p_obs.player_sum - 1,
                           e.p_act]
        Qcur  = p_feat.dot(e.theta)
        delta = Qcur - (r + Qnext) # Yes, in the gradient it's inverted.
        e.E.__iadd__(p_feat)

        # Note: Eligibility traces could be done slightly more succinctly by
        # scaling the feature vectors themselves. See Silver slides.
        e.theta.__isub__(e.alpha * delta * e.E)

        e.E.__imul__(e.lmbda)

    return e.set(p_obs=o, p_act=a), a


def wrapup(e, o, r):
    e, _ = think(e, o, r, done=True)
    return e.set(p_obs=None, p_act=None, E=np.zeros((36)))


def Q(e):
    return e.feature.dot(e.theta) # Matrix of products of each feature vector
                                  # with weights.
