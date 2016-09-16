# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
import pyrsistent

LinfaExperience = pyrsistent.immutable(
                    'feature_vec, theta, E, epsi, init_alpha, p_alpha, lmbda,'
                    ' p_obs, p_act, p_feat, act_space')


# pylint: disable=too-many-arguments
def init(lmbda, init_alpha, epsi, feature_vec, n_weights, act_space,
        theta=None):
    """

    Arguments:
        feature_vec - function mapping (observation, action) pairs to
                      FeatureVecs
        n_weights   - number of weights == length of the feature vectors
        act_space   - the OpenAI gym.spaces.discrete.Discrete action space of
                      the problem
    """
    if theta is None:
        theta = np.zeros(n_weights)

    return LinfaExperience(feature_vec=feature_vec,
                           theta=theta,
                           E=np.zeros(n_weights),
                           epsi=epsi,
                           init_alpha=init_alpha,
                           p_alpha=init_alpha,
                           lmbda=lmbda,
                           p_obs=None, # p … previous
                           p_act=None,
                           p_feat=None,
                           act_space=act_space)


def true_with_prob(p):
    return np.random.choice(2, p=[1-p, p])


def choose_action(e, o):
    if true_with_prob(e.epsi):
        return e.act_space.sample()
    else:
        return max(xrange(e.act_space.n),
                   key=lambda a: e.feature_vec(o, a).dot(e.theta))


def think(e, o, r, done=False):
    """

    Args:
        e … experience
        o … observation
        r … reward
    """

    if not done:
        a     = choose_action(e, o) # action
        feat  = e.feature_vec(o, a)
        Qnext = feat.dot(e.theta)
            # expected Q of next action

        # Credits: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        # Note: The paper doesn't mention the case when the previous feature
        # vector equals the current feature vector and gamma = 1. This case
        # would lead to a division by zero. We return p_alpha in this case.
        if e.p_feat:
            diffdot = abs(feat.alphabounds_diffdot(e.p_feat, e.E)) \
                          or 1.0/e.p_alpha
            alpha = min(e.p_alpha, 1.0 / diffdot)
        else:
            alpha = e.p_alpha
    else:
        a     = None
        feat  = None
        Qnext = 0
        alpha = e.p_alpha

    if e.p_obs is not None: # Except for first timestep.
        Qcur  = e.p_feat.dot(e.theta)
        delta = Qcur - (r + Qnext) # Yes, in the gradient it's inverted.
        e.p_feat.add_to(e.E)

        # Note: Eligibility traces could be done slightly more succinctly by
        # scaling the feature vectors themselves. See Silver slides.
        e.theta.__isub__(alpha * delta * e.E)

        e.E.__imul__(e.lmbda)

    return e.set(p_alpha=alpha, p_obs=o, p_act=a, p_feat=feat), a


def wrapup(e, o, r):
    e, _ = think(e, o, r, done=True)
    return e.set(p_obs=None, p_act=None, p_feat=None, E=np.zeros(e.E.shape))


def Q(e):
    return e.feature.dot(e.theta) # Matrix of products of each feature vector
                                  # with weights.
