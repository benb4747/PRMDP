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
