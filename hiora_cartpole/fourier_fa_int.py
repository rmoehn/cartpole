# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools

import numpy as np
import sympy as sp

def dot(es1, es2):
    return sum([e[0]*e[1] for e in zip(es1, es2)])


# Copied from hiora_cartpole.fourier_fa
def c_matrix(order, n_dims):
    """
    Generates the parameter (C) vectors for all terms in the Fourier FA.
    """

    # All entries from cartesian product {0, …, order+1}^n_dims.
    return np.array(
                list( itertools.product(range(order+1), repeat=n_dims) ),
                dtype=np.int32)


def sum_term(integral, c, c_vec):
    return integral.subs(zip(c, c_vec))


def make_sym_Q_s0(state_ranges, order):
    n_dims = state_ranges.shape[1]
    denorm_factor = np.prod( np.diff(state_ranges[:,1:], axis=0) )
        # We calculate the integral of the normalized fn over [0, 1]. Need to
        # denormalize afterwards.
    C = sp.symbols("c0:" + str(n_dims), integer=True)
    S = sp.symbols("s0:" + str(n_dims), real=True)

    integral = reduce(lambda f, s: sp.Integral(f, (s, 0, 1)),
                    S[1:], sp.cos(sp.pi * dot(S, C))).doit()

    c_vecs          = c_matrix(order, n_dims)
    sum_terms       = [sum_term(integral, C, c_vec) for c_vec in c_vecs]
    np_sum_terms    = [sp.lambdify(S[0], t, np) for t in sum_terms]

    def sym_Q_s0_inner(theta, a, s0):
        ns0 = (s0 - state_ranges[0][0]) \
                / (state_ranges[1][0] - state_ranges[0][0])
        theta_a = theta[a * c_vecs.shape[0]:(a+1) * c_vecs.shape[0]]
        return denorm_factor * \
            np.dot(theta_a, np.array([npst(ns0) for npst in np_sum_terms]))

    return sym_Q_s0_inner
