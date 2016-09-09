# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
import pyrsistent

LinfaExperience = pyrsistent.immutable(
                    'feature_vec, theta, E, epsi, alpha, lmbda, p_obs, p_act,' \
                    ' act_space')


# pylint: disable=too-many-arguments
def init(lmbda, alpha, epsi, feature_vec, n_weights, act_space):
    """

    Arguments:
        feature_vec - function mapping (observation, action) pairs to feature
                      vectors
        n_weights   - number of weights == length of the feature vectors
        act_space   - the OpenAI gym.spaces.discrete.Discrete action space of
                      the problem
    """
    return LinfaExperience(feature_vec=feature_vec,
                           theta=np.zeros(n_weights),
                           E=np.zeros(n_weights),
                           epsi=epsi,
                           alpha=alpha,
                           lmbda=lmbda,
                           p_obs=None, # p … previous
                           p_act=None,
                           act_space=act_space)


def true_with_prob(p):
    return np.random.choice(2, p=[1-p, p])


def choose_action(e, o):
    if true_with_prob(e.epsi):
        return e.act_space.sample()
    else:
        return max(xrange(e.act_space.n),
                   key=lambda a: e.feature_vec(o, a).dot(e.theta)[0])


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
    else:
        a     = None
        Qnext = 0

    if e.p_obs is not None: # Except for first timestep.
        p_feat = e.feature_vec(e.p_obs, e.p_act)
        Qcur  = p_feat.dot(e.theta)
        delta = Qcur - (r + Qnext) # Yes, in the gradient it's inverted.
        e.E[p_feat.indices] += 1.0

        # Note: Eligibility traces could be done slightly more succinctly by
        # scaling the feature vectors themselves. See Silver slides.
        e.theta.__isub__(e.alpha * delta * e.E)

        e.E.__imul__(e.lmbda)

    return e.set(p_obs=o, p_act=a), a


def wrapup(e, o, r):
    e, _ = think(e, o, r, done=True)
    return e.set(p_obs=None, p_act=None, E=np.zeros(e.E.shape))


def Q(e):
    return e.feature.dot(e.theta) # Matrix of products of each feature vector
                                  # with weights.
