import numpy as np
from scipy.stats import chi2, norm, poisson, binom
import time
import itertools as it

from math import log, exp
import matplotlib.pyplot as plt
from numba import njit

from .P_RMDP import P_RMDP
from .NP_RMDP import NP_RMDP


class NV_RMDP:
    def __init__(
        self,
        para,
        q,
        h,
        c,
        b,
        C,
        discount,
        P_0,
        dist,
        t_max,
        VI_tol,
        BS_tol,
        P_hat,
        distance,
        kappa,
        theta_0,
        theta_hat,
        alpha,
        M,
        N,
        timeout,
        solver_cores,
    ):

        self.para = para
        self.q = q
        self.h = h
        self.c = c
        self.b = b
        self.C = C
        self.discount = discount
        self.P_0 = P_0
        self.dist = dist
        self.t_max = t_max
        self.VI_tol = VI_tol
        self.BS_tol = BS_tol

        self.distance = distance
        self.kappa = kappa
        self.P_hat = P_hat

        self.S = C + 1
        self.A = C + 1

        self.theta_0 = theta_0
        self.MLE = theta_hat
        self.alpha = alpha
        self.M = M
        self.N = N
        self.timeout = timeout
        self.solver_cores = solver_cores

    def compute_rewards(self):
        r = np.zeros((self.S, self.A, self.S))
        for (s, a, s_) in it.product(range(self.S), range(self.A), range(self.S)):
            sold = max(min(s + a, self.C) - s_, 0)
            hold = min(s + a, self.C) - sold
            r[s, a, s_] = (
                self.q * sold - self.h * hold - self.c * a - self.b * int(s_ == 0)
            )
        if np.min(r) < 0:
            r = r + abs(np.min(r)) + 1
        self.r = r
        return

    def construct_MDP(self):
        if self.para == True:
            self.MDP = P_RMDP(
                range(self.S),
                range(self.A),
                self.r,
                self.discount,
                self.P_0,
                self.dist,
                self.t_max,
                self.VI_tol,
                self.BS_tol,
                self.theta_0,
                self.MLE,
                self.alpha,
                self.M,
                self.N,
                self.kappa,
                self.P_hat,
                self.timeout,
                self.solver_cores,
            )
        else:
            self.MDP = NP_RMDP(
                range(self.S),
                range(self.A),
                self.r,
                self.discount,
                self.P_0,
                self.P_hat,
                self.distance,
                self.kappa,
                self.t_max,
                self.VI_tol,
                self.BS_tol,
                self.timeout,
                self.solver_cores,
            )
        return self.MDP
