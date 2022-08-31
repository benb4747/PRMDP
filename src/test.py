import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, poisson, binom
import gurobipy as gp
from gurobipy import GRB
import time
import itertools as it

from math import log, exp
from scipy.optimize import bisect
from scipy.misc import derivative
import matplotlib.pyplot as plt
from numba import njit


@njit
def binom_coef(n, k):
    out = 1
    for i in range(1, k + 1):
        out *= (n + 1 - i) / i
    return out


@njit
def binompmf(n, p, x, log_pmf=False):
    if p == 0 and x > 0:
        return 0
    if p == 0 and x == 0:
        return 1
    if p == 1:
        return int(x == n)
    if x < 0 or x > n:
        return 0

    out = log(binom_coef(n, x))
    out = out + log(p) * x
    out = out + log(1 - p) * (n - x)

    if not log_pmf:
        out = exp(out)
    return out


@njit
def binom_conf_val(A, C, N, theta, MLE):
    val = 0
    for a in range(A):
        val += N * C * ((theta[a] - MLE[a]) ** 2) / (MLE[a] * (1 - MLE[a]))
    return val


"""
@njit
def probs_fast(L, i_range, i_max, p):
    probs = []
    N = np.shape(i_range)[0]
    for n in range(N):
        i = i_range[n]
        pr = 1
        for d in range(L):
            pr *= binompmf(i_max[d], p[d], i[d])
        probs.append(pr)

    return probs
"""


@njit
def binom_probs_fast(S, A, s, a, theta):
    probs = []
    C = S - 1
    for s_ in range(S):
        if s_ == 0:
            pr = 0
            for d in range(min(s + a, C), C + 1):
                pr += binompmf(C, theta, d)
        else:
            pr = binompmf(C, theta, min(s + a, C) - s_)
        probs.append(pr)
    return probs


@njit
def poisson_pmf(x, lam):
    fac = 1
    if x > 1:
        for i in range(2, x + 1):
            fac *= i
    return (exp(-lam) * lam**x) / fac


@njit
def poisson_probs_fast(S, A, s, a, lam):
    probs = []
    C = S - 1
    for s_ in range(S):
        if s_ == 0:
            pr = 1
            for d in range(min(s + a, C)):
                pr -= poisson_pmf(d, lam)
        else:
            pr = poisson_pmf(min(s + a, C) - s_, lam)
        probs.append(pr)
    return probs


class NP_RMDP:
    def __init__(
        self,
        states,
        actions,
        rewards,
        discount,
        P_0,
        P_hat,
        distance,
        kappa,
        t_max,
        eps,
        tol,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.discount = discount
        self.P_0 = P_0
        self.P_hat = P_hat
        self.distance = distance
        self.kappa = kappa
        self.t_max = t_max
        self.eps = eps
        self.tol = tol

        self.S = len(states)
        self.A = len(actions)

    def projection_obj_true(self, alpha, s, b, a, beta, zeta):
        return (
            -beta * alpha
            + zeta
            - sum(
                [
                    self.P_hat[s, a, s_]
                    * (max(1 + (-alpha * b[s_] + zeta) / 2, 0) ** 2 - 1)
                    for s_ in range(self.S)
                ]
            )
        )

    def projection_obj(
        self, alpha, s, b, a, beta, S_hat=-1, case=0, S_=0, states_nz=[]
    ):
        if self.distance == "KLD":
            return -beta * alpha - log(
                sum(
                    [self.P_hat[s, a, s_] * exp(-alpha * b[s_]) for s_ in range(self.S)]
                )
            )

        if self.distance == "mchisq":
            if S_hat != -1:
                P_hat = self.P_hat[s, a, states_nz]
                if S_hat == S_:
                    return -1
                b = np.array(b)
                if case == 0:
                    zeta = -2 + alpha * b[S_hat]
                elif case == 1:
                    zeta = -2 + alpha * b[S_hat - 1]
                elif case == 2:
                    zeta = (
                        2 * sum(P_hat[:S_hat]) + alpha * sum((b * P_hat)[S_hat:])
                    ) / sum(P_hat[S_hat:])

                return (
                    -beta * alpha
                    + zeta
                    + sum(P_hat[:S_hat])
                    - sum(
                        (
                            P_hat
                            * (-alpha * b + zeta + (1 / 4) * (-alpha * b + zeta) ** 2)
                        )[S_hat:]
                    )
                )

    def mchisq_alpha_sol(self, s, b, a, beta, S_hat=0, case=0, S_=0, states_nz=[]):
        P_hat = self.P_hat[s, a, states_nz]
        b = np.array(b)
        if case == 0:
            x = b[S_hat]
            sol = 2 * (-beta + x) / sum((P_hat * (x - b) ** 2)[S_hat:])
        elif case == 1:
            x = b[S_hat - 1]
            sol = 2 * (-beta + x) / sum((P_hat * (x - b) ** 2)[S_hat:])

        elif case == 2:
            if S_hat == S_ - 1:
                sol = 0
            else:
                x = sum((b * P_hat)[S_hat:]) / sum(P_hat[S_hat:])
                sol = 2 * (-beta + x) / sum((P_hat * (x - b) ** 2)[S_hat:])

        return sol

    def mchisq_alpha_bound(self, s, b, a, beta, S_hat=0, case=0, S_=0, states_nz=[]):
        P_hat = self.P_hat[s, a, states_nz]
        if case == 0:
            lb = 2 / sum(((b[S_hat] - b) * P_hat)[S_hat:])
            return [lb]
        elif case == 1:
            lb = 0
            ub = 2 / sum(((b[S_hat - 1] - b) * P_hat)[S_hat:])
            return [ub]
        elif case == 2:
            lb = 2 / sum(((b[S_hat - 1] - b) * P_hat)[S_hat:])
            if S_hat != S_ - 1:
                ub = 2 / sum(((b[S_hat] - b) * P_hat)[S_hat:])
                return [lb, ub]
            return [lb]

    def distance_term(self, P, Q):
        # gives Q * phi(P / Q)
        if Q == 0 and P == 0:
            return 0
        if self.distance == "KLD":
            return P * log(P / Q)
        if self.distance == "mchisq":
            return (P - Q) ** 2 / Q

    def projection_deriv(
        self, alpha, s, b, a, beta, S_hat=0, case=0, S_=0, states_nz=[]
    ):
        return derivative(
            func=self.projection_obj,
            x0=alpha,
            args=(s, b, a, beta, S_hat, case, S_, states_nz),
            dx=1e-6,
        )

    def solve_projection(self, s, b, a, beta, delta, method="sp"):
        if sum(self.P_hat[s, a] * b) <= beta:
            # print("trivial solution is optimal.")
            dist = sum(
                [
                    self.distance_term(self.P_hat[s, a, s_], self.P_hat[s, a, s_])
                    for s_ in range(self.S)
                ]
            )
            return [dist, dist]
        if self.distance == "KLD":
            l = 0
            if np.min(self.P_hat) == 0:
                m = 1e-6
            else:
                m = np.min(self.P_hat)
            u = log(1 / np.min(m)) * (1 / (beta - min(b)))
            i = 1

            condition = True
            while condition:
                x = (l + u) / 2
                try:
                    self.projection_deriv(x, s, b, a, beta)
                except ValueError:
                    u = x
                    continue

                if self.projection_deriv(x, s, b, a, beta) > 0:
                    l = x
                else:
                    u = x
                if u - l <= delta / max(b):
                    condition = False
                i = i + 1

            d_bar = [
                self.projection_obj(l, s, b, a, beta),
                self.projection_obj(l, s, b, a, beta) + max(b) * (u - l),
            ]
            d_bar = [min(d_bar), max(d_bar)]

            return d_bar

        elif self.distance == "mchisq":
            if method == "proj_qp":
                m = gp.Model()
                m.Params.LogToConsole = 0
                m.Params.OutputFlag = 0
                m.Params.Threads = 1
                m.Params.Presolve = 0

                zeta = m.addVar(name="eta", lb=-GRB.INFINITY)
                alpha = m.addVar(name="alpha")
                z = m.addVars(range(self.S), lb=0, name="z")
                x = m.addVars(range(self.S), lb=-GRB.INFINITY, name="x")
                keys = range(self.S)
                m.addConstrs(z[s_] >= -alpha * b[s_] + zeta + 2 for s_ in keys)
                m.addConstrs(x[s_] >= 0.25 * z[s_] * z[s_] - 1 for s_ in keys)

                Obj = (
                    -beta * alpha
                    + zeta
                    - gp.quicksum(self.P_hat[s, a, s_] * x[s_] for s_ in keys)
                )

                m.setObjective(Obj, GRB.MAXIMIZE)

                m.optimize()

                alpha_star = alpha.x
                zeta_star = zeta.x

                del m

                return [
                    self.projection_obj_true(alpha_star, s, b, a, beta, zeta_star),
                    self.projection_obj_true(alpha_star, s, b, a, beta, zeta_star),
                ]
            elif method == "proj_sort":

                states_nz = [s_ for s_ in range(self.S) if self.P_hat[s, a, s_] != 0]
                states_nz = list(
                    reversed(sorted(range(len(states_nz)), key=lambda k: b[k]))
                )
                S_ = len(states_nz)
                b_sorted = b[states_nz]
                alpha_sols = np.zeros(S_, dtype="object")
                best_objs = []
                for S_hat in range(S_ + 1):
                    if S_hat == S_:
                        best_alpha = [(0, 0, S_hat)]
                        best_objs.append(-1)
                        continue
                    else:
                        best_alpha = []
                    for case in range(3):
                        if ((case == 0) and (S_hat == S_ - 1)) or (
                            (case == 1) and S_hat == 0
                        ):
                            continue
                        alpha_sol = self.mchisq_alpha_sol(
                            s, b_sorted, a, beta, S_hat, case, S_, states_nz
                        )
                        bounds = self.mchisq_alpha_bound(
                            s, b_sorted, a, beta, S_hat, case, S_, states_nz
                        )
                        if case == 0:
                            lb = bounds[0]
                            best = max(alpha_sol, lb)
                        else:
                            if len(bounds) == 2:
                                lb, ub = bounds
                            else:
                                lb = bounds[0]

                            if alpha_sol < lb:
                                best = lb
                            elif alpha_sol > ub:
                                best = ub
                            else:
                                best = alpha_sol

                        best_alpha.append((best, case, S_hat))
                    objs = np.array(
                        [
                            self.projection_obj(
                                alpha, s, b_sorted, a, beta, S_hat, case, S_, states_nz
                            )
                            for (alpha, case, S_hat) in best_alpha
                        ]
                    )
                    # print("S_hat = ", S_hat, ", objs = ", objs)
                    alpha_sols[S_hat] = tuple(best_alpha[np.argmax(objs)])
                    best_objs.append(max(objs))

                return [max(best_objs), max(best_objs)]

    def Bellman_update(self, v, s, t, method):
        # print("---- Bellman update step t = %s for s = %s----\n" %(t,s))

        if "proj" in method:
            R_bar = np.max(self.rewards) / (1 - self.discount)
            delta = self.tol * self.kappa / (2 * self.A * R_bar + self.A * self.tol)
            l = max(
                [
                    min(
                        [
                            self.rewards[s, a, s_] + self.discount * v[s_]
                            for s_ in range(self.S)
                        ]
                    )
                    for a in range(self.A)
                ]
            )
            u = R_bar
            alpha = np.zeros(self.A)
            condition = True
            while condition:
                # print("l = ", l, ", u = ", u, "\n")
                theta = (l + u) / 2
                d_bar = np.zeros((self.A, 2))
                for a in range(self.A):
                    b = np.array(self.rewards[s, a]) + self.discount * np.array(v)
                    # print("b = ", b, "theta = ", theta)
                    d_bar[a] = self.solve_projection(s, b, a, theta, delta, method)
                # print(d_bar)
                if sum(d_bar[:, 1]) <= self.kappa:
                    u = theta
                elif sum(d_bar[:, 0]) > self.kappa:
                    l = theta

                if (u - l <= self.tol) or (
                    self.kappa >= sum(d_bar[:, 0]) and self.kappa < sum(d_bar[:, 1])
                ):
                    condition = False

            return theta, alpha

        elif method == "QP":
            obj = self.find_policy(s, v)[-1]
            return obj, 0

    def Bellman_obj(self, v, pi, P):
        return [
            sum(
                [
                    pi[s, a]
                    * sum(
                        [
                            P[s, a, s_]
                            * (self.rewards[s, a, s_] + self.discount * v[s_])
                            for s_ in range(self.S)
                        ]
                    )
                    for a in range(self.A)
                ]
            )
            for s in range(self.S)
        ]

    def value_iteration(self, method="sp"):
        t = 0
        v_new = np.zeros(self.S)
        pi_star = np.zeros((self.S, self.A))
        Delta = 10000
        alpha = np.zeros((self.S, self.A))
        worst = np.zeros((self.S, self.A, self.S))
        while (
            Delta >= self.eps * (1 - self.discount) / (2 * self.discount)
        ) and t < self.t_max:
            if t == 0:
                Delta = 0
            v = v_new.copy()
            for s in range(self.S):
                v_new[s], alpha[s] = self.Bellman_update(v, s, t, method)
            Delta = max(np.array(v_new) - np.array(v))
            t += 1
        self.values = v_new

        for s in range(self.S):
            pi_star[s], worst[s] = self.find_policy(s, v_new)[:2]

        obj = np.matmul(np.array(self.P_0), np.array(v_new))

        return {
            "Policy": np.round(pi_star, 4),
            "Values": np.round(v_new, 4),
            "Objective": np.round(obj, 4),
            "Worst": np.round(worst, 4),
        }

    def find_policy(self, s, v):
        m = gp.Model()
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.Presolve = 1

        pi = m.addVars(range(self.A), vtype=GRB.CONTINUOUS, name="pi")
        nu = m.addVars(range(self.A), name="nu", lb=-GRB.INFINITY)
        eta = m.addVar(name="eta")
        z = m.addVars(
            [(a, s_) for a in range(self.A) for s_ in range(self.S)], lb=0, name="z"
        )
        x = m.addVars(
            [(a, s_) for a in range(self.A) for s_ in range(self.S)], lb=0, name="x"
        )
        b = np.zeros((self.A, self.S))
        keys = x.keys()
        for (a, s_) in keys:
            b[a, s_] = self.rewards[s, a, s_] + self.discount * v[s_]

        m.addConstr(gp.quicksum(pi) == 1)

        if self.distance == "KLD":

            log_x = m.addVars(
                [(a, s_) for a in range(self.A) for s_ in range(self.S)], name="log_x"
            )
            log_eta = m.addVar(name="log_eta")
            m.addConstrs(x[a, s_] == eta + z[a, s_] for (a, s_) in keys)

            for (a, s_) in keys:
                m.addGenConstrLog(x[a, s_], log_x[a, s_])
            m.addGenConstrLog(eta, log_eta)

            m.addConstrs(
                eta * (log_eta - log_x[a, s_]) <= pi[a] * b[a, s_] - nu[a]
                for (a, s_) in keys
            )

        elif self.distance == "mchisq":
            m.addConstrs(
                z[a, s_] >= nu[a] - pi[a] * b[a, s_] + 2 * eta for (a, s_) in keys
            )
            x1 = m.addVars(keys, lb=-GRB.INFINITY, name="x1")
            x2 = m.addVars(keys, lb=-GRB.INFINITY, name="x2")
            x2_abs = m.addVars(keys, lb=0, name="x2_abs")

            m.addConstrs(x1[a, s_] == 0.5 * (eta - x[a, s_]) for (a, s_) in keys)
            m.addConstrs(x2[a, s_] == 0.5 * (eta + x[a, s_]) for (a, s_) in keys)
            m.addConstrs(x2_abs[a, s_] == gp.abs_(x2[a, s_]) for (a, s_) in keys)

            m.addConstrs(
                z[a, s_] * z[a, s_] + x1[a, s_] * x1[a, s_]
                <= x2_abs[a, s_] * x2_abs[a, s_]
                for (a, s_) in keys
            )

        Obj = (
            gp.quicksum(nu)
            + eta * (self.A - self.kappa)
            - 0.25 * gp.quicksum(self.P_hat[s, a, s_] * x[a, s_] for (a, s_) in keys)
        )

        m.setObjective(Obj, GRB.MAXIMIZE)
        m.optimize()

        pi_star = np.array(m.getAttr("x", pi).values())
        nu_star = np.array(m.getAttr("x", nu).values())
        eta_star = eta.x

        y = np.zeros((self.A, self.S))
        worst = np.zeros((self.A, self.S))

        for (a, s_) in it.product(range(self.A), range(self.S)):
            y[a, s_] = (nu_star[a] - pi_star[a] * b[a, s_]) / eta_star
            if self.distance == "KLD":
                worst[a, s_] = self.P_hat[s, a, s_] * exp(y[a, s_])
            elif self.distance == "mchisq":
                worst[a, s_] = self.P_hat[s, a, s_] * max(1 + y[a, s_] / 2, 0)
        obj = m.ObjVal

        del m

        return pi_star, worst, obj


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
        eps,
        policy,
        P_hat,
        distance,
        kappa,
        tol,
        theta_0,
        theta_hat,
        alpha,
        gap,
        N,
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
        self.eps = eps

        self.distance = distance
        self.kappa = kappa
        self.tol = tol
        self.P_hat = P_hat

        self.S = C + 1
        self.A = C + 1

        self.theta_0 = theta_0
        self.MLE = theta_hat
        self.alpha = alpha
        self.gap = gap
        self.N = N

    def compute_rewards(self):
        r = np.zeros((self.S, self.A, self.S))
        for (s, a, s_) in it.product(range(self.S), range(self.A), range(self.S)):
            sold = max(min(s + a, self.C) - s_, 0)
            hold = min(s + a, self.C) - sold
            r[s, a, s_] = (
                1000
                + self.q * sold
                - self.h * hold
                - self.c * a
                - self.b * int(s_ == 0)
            )
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
                self.eps,
                self.theta_0,
                self.MLE,
                self.alpha,
                self.gap,
                self.N,
                self.tol,
                self.kappa,
                self.P_hat,
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
                self.eps,
                self.tol,
            )
        return self.MDP


class P_RMDP:
    def __init__(
        self,
        states,
        actions,
        rewards,
        discount,
        P_0,
        dist,
        t_max,
        eps,
        theta_0,
        theta_hat,
        alpha,
        gap,
        N,
        tol,
        kappa,
        P_hat,
    ):

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.discount = discount
        self.P_0 = P_0
        self.MLE = theta_hat
        self.alpha = alpha
        self.dist = dist
        self.t_max = t_max
        self.eps = eps
        self.gap = gap
        self.N = N
        self.tol = tol
        self.kappa = kappa
        self.P_hat = P_hat

        self.S = len(states)
        self.A = len(actions)

    def construct_conf(self):
        theta_base = []
        if self.dist == "binomial":
            theta_base = []
            gap = self.gap
            ranges = np.zeros((self.S, self.A), dtype="object")
            for (s, a) in it.product(range(self.S), range(self.A)):
                th = self.MLE[s, a]
                ub = min(
                    th
                    + np.sqrt(
                        (th * (1 - th) * chi2.ppf(1 - self.alpha, self.A))
                        / (self.N * (self.S - 1))
                    ),
                    1,
                )
                lb = max(
                    th
                    - np.sqrt(
                        (th * (1 - th) * chi2.ppf(1 - self.alpha, self.A))
                        / (self.N * (self.S - 1))
                    ),
                    0,
                )
                ranges[s, a] = np.arange(lb, ub + gap, gap)

            theta_base = [
                it.product(*[ranges[s, a] for a in range(self.A)])
                for s in range(self.S)
            ]
            Theta = np.zeros(self.S, dtype="object")
            for s in range(self.S):
                Theta_s = [
                    theta
                    for theta in theta_base[s]
                    if binom_conf_val(self.A, self.S - 1, self.N, theta, self.MLE[s])
                    <= chi2.ppf(1 - self.alpha, self.A)
                ]
                if tuple(self.MLE[s]) not in Theta_s:
                    Theta_s.append(tuple(self.MLE[s]))
                Theta[s] = Theta_s
            self.Theta = Theta

        self.num_params = [len(self.Theta[s]) for s in range(self.S)]

    def compute_probs(self):
        if self.dist in ["Binomial", "binomial"]:
            probs = np.zeros((self.S, self.A), dtype="object")
            for (s, a) in it.product(range(self.S), range(self.A)):
                pr = np.zeros((self.num_params[s], self.S))
                for i in range(self.num_params[s]):
                    pr[i] = binom_probs_fast(self.S, self.A, s, a, self.Theta[s][i][a])
                probs[s, a] = pr

            self.probs = probs

    def Bellman_LP(self, v, s, t, AS):
        m = gp.Model(name="v_{%s,%s}" % (s, t))
        dummy = m.addVar(vtype=GRB.CONTINUOUS, name="dummy", lb=-GRB.INFINITY)
        pi = m.addVars(range(self.A), vtype=GRB.CONTINUOUS, name="pi")

        m.addConstr(gp.quicksum(pi) == 1)
        m.Params.LogToConsole = 0
        m.Params.OutputFlag = 0
        m.Params.Threads = 1

        Theta = [tuple(theta) for theta in self.Theta[s]]

        reward = np.array(
            [
                [
                    sum(
                        [
                            self.probs[s, a][Theta.index(theta), s_]
                            * (self.rewards[s, a, s_] + self.discount * v[s_])
                            for s_ in range(self.S)
                        ]
                    )
                    for theta in AS
                ]
                for a in range(self.A)
            ]
        )

        m.addConstrs(
            dummy <= gp.quicksum(pi[a] * reward[a, i] for a in range(self.A))
            for i in range(len(AS))
        )

        dv = pi

        m.setObjective(dummy, GRB.MAXIMIZE)

        m.optimize()

        return m, dv, dummy, reward

    def projection_obj(self, theta, s, b, a):
        return (self.N * (self.S - 1) * (theta - self.MLE[s, a]) ** 2) / (
            self.MLE[s, a] * (1 - self.MLE[s, a])
        )

    def projection_deriv(self, theta, s, a):
        return (2 * self.N * (self.S - 1) * (theta - self.MLE[s, a])) / (
            self.MLE[s, a] * (1 - self.MLE[s, a])
        )

    def solve_projection(self, s, b, a, beta, delta, root_gap=0.01, beta_tol=1e-6):
        if sum(self.P_hat[s, a] * b) <= beta:
            return [self.MLE[s, a], 0]
        l, u = 0, 1
        # find root intervals
        if self.dist == "binomial":
            theta_min = 0
            theta_max = 1
        theta = self.MLE[s, a]
        EV = beta
        intervals = []
        for run in range(2):
            while EV >= beta:
                if theta <= theta_min or theta >= theta_max:
                    continue
                if run == 0:
                    theta -= root_gap
                else:
                    theta += root_gap
                EV = sum(np.array(binom_probs_fast(self.S, self.A, s, a, theta)) * b)
            if run == 0:
                intervals.append((theta, theta + root_gap))
            else:
                intervals.append((theta - root_gap, theta))

        roots = []
        if self.dist == "binomial":
            delta_ = np.sqrt(
                delta * self.MLE[s, a] * (1 - self.MLE[s, a]) / (self.N * self.S - 1)
            )
        # print("Running bisection for proj, state ", s, "action ", a)
        for run in range(len(intervals)):
            # print("Running beta part of run ", run)
            condition = True
            l, u = intervals[run]
            while condition:
                x = (l + u) / 2
                if self.dist == "binomial":
                    P = np.array(binom_probs_fast(self.S, self.A, s, a, x))
                if sum(P * b) <= beta and u - l <= delta_:
                    condition = False
                elif sum(P * b) < beta:
                    l = x
                else:
                    u = x
            roots.append((l, u))

        objs = [self.projection_obj((u + l) / 2, s, b, a) for (l, u) in roots]
        l, u = roots[np.argmin(objs)]
        opt = (l + u) / 2
        objs_opt = [self.projection_obj(sol, s, b, a) for sol in [l, u]]
        return opt, np.array([min(objs_opt), max(objs_opt)])

    def Bellman_BS(self, v, s, t):
        # print("---- Bellman update step t = %s for s = %s----\n" %(t,s))

        R_bar = np.max(self.rewards) / (1 - self.discount)
        delta = self.tol * self.kappa / (2 * self.A * R_bar + self.A * self.tol)
        l = max(
            [
                min(
                    [
                        self.rewards[s, a, s_] + self.discount * v[s_]
                        for s_ in range(self.S)
                    ]
                )
                for a in range(self.A)
            ]
        )
        u = R_bar
        conf = np.zeros((self.A, 2))
        condition = True
        while condition:
            # print("l = ", l, ", u = ", u, "\n")
            x = (l + u) / 2
            theta_BS = np.zeros(self.A)
            for a in range(self.A):
                b = np.array(self.rewards[s, a]) + self.discount * np.array(v)
                # print("b = ", b, "theta = ", theta)
                theta_BS[a], conf[a] = self.solve_projection(s, b, a, x, delta)
            # print(d_bar)

            if sum(conf[:, 1]) <= self.kappa:
                u = x
            elif sum(conf[:, 0]) > self.kappa:
                l = x

            if (u - l <= self.tol) or (
                self.kappa < sum(conf[:, 1]) and self.kappa >= sum(conf[:, 0])
            ):
                condition = False

        return x, theta_BS

    def Bellman_CS(self, v, s, t, CS_tol=1e-6, k_max=100):
        theta_0 = self.Theta[s][0]
        Theta_k = [tuple(theta_0)]
        theta_k = Theta_k[0]
        Theta = [tuple(theta) for theta in self.Theta[s]]
        k = 0
        while k < k_max:
            if k == 0:
                m, dv, dummy = self.Bellman_LP(v, s, t, Theta_k)[:3]
            else:
                i = Theta.index(theta_k)
                reward = np.array(
                    [
                        [
                            sum(
                                [
                                    self.probs[s, a][i, s_]
                                    * (self.rewards[s, a, s_] + self.discount * v[s_])
                                    for s_ in range(self.S)
                                ]
                            )
                        ]
                        for a in range(self.A)
                    ]
                )
                pi = dv

                m.addConstr(
                    dummy <= gp.quicksum(pi[a] * reward[a] for a in range(self.A))
                )
                m.optimize()

                pi = dv
                pi_k = np.array((m.getAttr("x", pi).values()))
                reward_all = np.array(
                    [
                        [
                            sum(
                                [
                                    self.probs[s, a][i, s_]
                                    * (self.rewards[s, a, s_] + self.discount * v[s_])
                                    for s_ in range(self.S)
                                ]
                            )
                            for i in range(self.num_params[s])
                        ]
                        for a in range(self.A)
                    ]
                )
                exp_reward = [
                    sum([pi_k[a] * reward_all[a, i] for a in range(self.A)])
                    for i in range(self.num_params[s])
                ]

            obj_k = m.ObjVal

            theta_k = tuple(self.Theta[s][np.argmin(exp_reward)])
            worst_obj = min(exp_reward)

            repeat = theta_k in Theta_k

            if not repeat:
                Theta_k.append(theta_k)

            else:
                reason = "repeat"
                break

            if abs(worst_obj - obj_k) <= CS_tol / 2 and not repeat:
                reason = "optimal"
                break

            else:
                k += 1

        if k == k_max and reason == "":
            reason = "k"

        return m, dv

    def Bellman_obj(self, v, pi, P):
        return [
            sum(
                [
                    pi[s, a]
                    * sum(
                        [
                            P[s, a, s_]
                            * (self.rewards[s, a, s_] + self.discount * v[s_])
                            for s_ in range(self.S)
                        ]
                    )
                    for a in range(self.A)
                ]
            )
            for s in range(self.S)
        ]

    def value_iteration(self, method="LP"):
        t = 0
        v_new = np.zeros(self.S)
        pi_star = np.zeros((self.S, self.A))
        Delta = 10000
        worst_param = np.zeros((self.S, self.A))
        worst_dist = np.zeros((self.S, self.A, self.S))
        start = time.perf_counter()
        reasons = []
        theta_BS = np.zeros((self.S, self.A))
        while (
            Delta >= self.eps * (1 - self.discount) / (2 * self.discount)
        ) and t < self.t_max:
            # print("t=%s\n" %t)
            if t == 0:
                Delta = 0
            v = v_new.copy()
            for s in range(self.S):
                AS = [tuple(theta) for theta in self.Theta[s]]
                if method == "LP":
                    m, dv, dummy, reward = self.Bellman_LP(v, s, t, AS)
                    obj = m.ObjVal

                elif method == "CS":
                    m, dv = self.Bellman_CS(v, s, t)
                    obj = m.ObjVal

                elif method == "BS":
                    obj, theta = self.Bellman_BS(v, s, t)
                    theta_BS[s] = theta

                if method in ["LP", "CS"]:
                    pi = dv
                    pi_star[s] = np.array((m.getAttr("x", pi).values()))
                    del m

                v_new[s] = obj

                Delta = max(np.array(v_new) - np.array(v))
            # print("ITER: ", t, "VALUES: ", v_new, "Delta: ", Delta,
            #      "Stopping: ", self.eps * (1 - self.discount) / (2 * self.discount))
            t += 1

        if method == "BS":
            for s in range(self.S):
                m, pi = self.Bellman_CS(v, s, t)
                pi_star[s] = np.array((m.getAttr("x", pi).values()))
                del m

        reward = np.array(
            [
                [
                    sum(
                        [
                            self.probs[s, a][i, s_]
                            * (self.rewards[s, a, s_] + self.discount * v[s_])
                            for s_ in range(self.S)
                        ]
                    )
                    for i in range(self.num_params[s])
                ]
                for a in range(self.A)
            ]
        )
        rewards_opt = [
            sum([pi_star[s, a] * reward[a, i] for a in range(self.A)])
            for i in range(self.num_params[s])
        ]
        for s in range(self.S):
            worst_ind = np.argmin(rewards_opt)
            worst_param[s] = np.array(self.Theta[s][worst_ind])
            for a in range(self.A):
                worst_dist[s, a] = self.probs[s, a][worst_ind]

        end = time.perf_counter()
        self.tt_opt = np.round(end - start, 3)
        self.dv = pi_star
        self.value_fcn = v_new
        self.obj = np.matmul(np.array(self.P_0), np.array(self.value_fcn))
        self.worst_dist = worst_dist

        if method == "CS":
            self.CS_reasons = reasons
        # print("Finished value iteration in %s secs and t=%s iterations."%(self.tt_opt, t))
        # print("Final values: ", v_new)

        return {
            "Policy": pi_star,
            "Values": v_new,
            "Objective": obj,
            "Worst param": worst_param,
            "Worst dist": worst_dist,
        }
