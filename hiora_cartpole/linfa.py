# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools
import operator

import numpy as np
import pyrsistent

#### Choosing actions

def true_with_prob(p):
    return np.random.choice(2, p=[1-p, p])


def choose_greedy(e, o):
    feats = [e.feature_vec(o, a) for a in xrange(e.act_space.n)]
    Qs    = [f.dot(e.theta) for f in feats]

    if all(Q == Qs[0] for Q in Qs):
        a = np.random.choice(2)
        return Qs[a], a, feats[a]
    else:
        return max(itertools.izip(Qs, xrange(e.act_space.n), feats),
                key=operator.itemgetter(0))


# Note: This does not follow the definition of an epsi-greedy strategy given in
# Sutton & Barto, 09/2016, p. 127. There, epsi is divided by the number of
# available actions.
def choose_action_Q(e, o):
    g_Q, g_a, g_feat = choose_greedy(e, o)

    if not true_with_prob(e.epsi):
        return g_Q, g_a, g_feat
    else:
        epsi_a = e.act_space.sample()
        return g_Q, epsi_a, e.feature_vec(o, epsi_a)


# Note: Despite this and the Q-learning one being quite similar, I can't unify
# them without making Sarsa slower.
def choose_action_Sarsa(e, o):
    if not true_with_prob(e.epsi):
        return choose_greedy(e, o)
    else:
        epsi_a    = e.act_space.sample()
        epsi_feat = e.feature_vec(o, epsi_a)
        return epsi_feat.dot(e.theta), epsi_a, epsi_feat


#### Main

# Note: I realize that this is getting unwieldy. At some point I should turn
# this into a learner object that modifies itself, but returns stuff that is not
# modified. Maybe.

LinfaExperience = pyrsistent.immutable(
                    'feature_vec, theta, E, epsi, init_alpha, p_alpha, lmbda,'
                    ' gamma, p_obs, p_act, p_feat, act_space,'
                    ' is_use_alpha_bounds, map_obs, choose_action')

# pylint: disable=too-many-arguments
def init(lmbda, init_alpha, epsi, feature_vec, n_weights, act_space,
        theta=None, is_use_alpha_bounds=False, map_obs=lambda x: x,
        choose_action=choose_action_Sarsa, gamma=1.0):
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
                           gamma=gamma,
                           p_obs=None, # p … previous
                           p_act=None,
                           p_feat=None,
                           act_space=act_space,
                           is_use_alpha_bounds=is_use_alpha_bounds,
                           map_obs=map_obs,
                           choose_action=choose_action)


# Note: This is based on Sutton & Barto 10/2015, p. 211. The 09/2016 version
# suddenly does this quite differently (p. 280).
def think(e, o, r, done=False):
    """

    Args:
        e … experience
        o … observation
        r … reward
    """

    o = e.map_obs(o)

    if not done:
        Qnext, a, feat = e.choose_action(e, o)

        # Credits: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        # Note: The paper doesn't mention the case when the previous feature
        # vector equals the current feature vector and gamma = 1. This case
        # would lead to a division by zero. We return p_alpha in this case.
        if e.is_use_alpha_bounds and e.p_feat:
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
        delta = Qcur - (r + e.gamma * Qnext) # Yes, in the gradient it's inverted.
        e.p_feat.add_to(e.E)

        # Note: Eligibility traces could be done slightly more succinctly by
        # scaling the feature vectors themselves. See Silver slides.
        e.theta.__isub__(alpha * delta * e.E)

        e.E.__imul__(e.lmbda)

    return e.set(p_alpha=alpha, p_obs=o, p_act=a, p_feat=feat), a


def wrapup(e, o, r, done=False):
    if done:
        o    = e.map_obs(o)
        e, _ = think(e, o, r, done=True)

    return e.set(p_obs=None, p_act=None, p_feat=None, E=np.zeros(e.E.shape))


def Q(e):
    return e.feature.dot(e.theta) # Matrix of products of each feature vector
                                  # with weights.
