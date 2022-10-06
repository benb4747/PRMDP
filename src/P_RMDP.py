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

from .njit_functions import *


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
        VI_tol,
        BS_tol,
        theta_0,
        theta_hat,
        alpha,
        M,
        N,
        kappa,
        P_hat,
        timeout,
        solver_cores,
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
        self.VI_tol = VI_tol
        self.BS_tol = BS_tol
        self.M = M
        self.N = N
        self.kappa = kappa
        self.P_hat = P_hat

        self.S = len(states)
        self.A = len(actions)
        self.timeout = timeout
        self.solver_cores = solver_cores

    def construct_conf(self):
        start = time.perf_counter()
        theta_base = []
        if self.dist == "binomial":
            theta_base = []
            M = self.M
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
                ranges[s, a] = np.array(
                    [lb + m * (ub - lb) / (M - 1) for m in range(M)]
                )

            theta_base = [
                it.product(*[ranges[s, a] for a in range(self.A)])
                for s in range(self.S)
            ]
            now = time.perf_counter()
            if now - start > self.timeout:
                return "T.O."
            Theta = np.zeros(self.S, dtype="object")
            kappa = chi2.ppf(1 - self.alpha, self.A)
            for s in range(self.S):
                Theta_s = []
                for theta in theta_base[s]:
                    now = time.perf_counter()
                    if now - start > self.timeout:
                        return "T.O."
                    if max(theta) > 1 or min(theta) < 0:
                        continue
                    if (
                        binom_conf_val(self.A, self.S - 1, self.N, theta, self.MLE[s])
                        <= kappa
                    ):
                        Theta_s.append(tuple(theta))
                # Theta_s = compute_conf_set(
                #     np.array(theta_base[s]),
                #     self.A,
                #     self.S - 1,
                #     self.N,
                #     self.MLE[s],
                #     kappa,
                # )
                if tuple(self.MLE[s]) not in Theta_s:
                    Theta_s.append(tuple(self.MLE[s]))
                Theta[s] = Theta_s
            self.Theta = Theta

        if self.dist == "poisson":
            theta_base = []
            M = self.M
            ranges = np.zeros((self.S, self.A), dtype="object")
            for (s, a) in it.product(range(self.S), range(self.A)):
                th = self.MLE[s, a]
                ub = th + np.sqrt((th * chi2.ppf(1 - self.alpha, self.A) / self.N))
                lb = max(
                    th - np.sqrt((th * chi2.ppf(1 - self.alpha, self.A) / self.N)),
                    0,
                )
                ranges[s, a] = np.array(
                    [lb + m * (ub - lb) / (M - 1) for m in range(M)]
                )

            theta_base = [
                it.product(*[ranges[s, a] for a in range(self.A)])
                for s in range(self.S)
            ]
            now = time.perf_counter()
            if now - start > self.timeout:
                return "T.O."
            Theta = np.zeros(self.S, dtype="object")
            kappa = chi2.ppf(1 - self.alpha, self.A)
            for s in range(self.S):
                Theta_s = []
                for theta in theta_base[s]:
                    now = time.perf_counter()
                    if now - start > self.timeout:
                        return "T.O."
                    if poisson_conf_val(self.A, self.N, theta, self.MLE[s]) <= kappa:
                        Theta_s.append(tuple(theta))
                # Theta_s = compute_conf_set(
                #     np.array(theta_base[s]),
                #     self.A,
                #     self.S - 1,
                #     self.N,
                #     self.MLE[s],
                #     kappa,
                # )
                if tuple(self.MLE[s]) not in Theta_s:
                    Theta_s.append(tuple(self.MLE[s]))
                Theta[s] = Theta_s
            self.Theta = Theta

        self.num_params = [len(self.Theta[s]) for s in range(self.S)]
        self.t_conf = np.round(time.perf_counter() - start, 3)
        return True

    def compute_probs(self):
        start = time.perf_counter()
        if self.dist in ["Binomial", "binomial"]:
            probs = np.zeros((self.S, self.A), dtype="object")
            for (s, a) in it.product(range(self.S), range(self.A)):
                now = time.perf_counter()
                if (now - start) + self.t_conf > self.timeout:
                    return "T.O."
                pr = np.zeros((self.num_params[s], self.S))
                for i in range(self.num_params[s]):
                    pr[i] = binom_probs_fast(self.S, self.A, s, a, self.Theta[s][i][a])
                probs[s, a] = pr

            self.probs = probs

            return True

        if self.dist in ["Poisson", "poisson"]:
            probs = np.zeros((self.S, self.A), dtype="object")
            for (s, a) in it.product(range(self.S), range(self.A)):
                now = time.perf_counter()
                if (now - start) + self.t_conf > self.timeout:
                    return "T.O."
                pr = np.zeros((self.num_params[s], self.S))
                for i in range(self.num_params[s]):
                    pr[i] = poisson_probs_fast(
                        self.S, self.A, s, a, self.Theta[s][i][a]
                    )
                probs[s, a] = pr

            self.probs = probs

            return True

    def Bellman_LP(self, v, s, t, AS, tt, method="LP"):
        start = time.perf_counter()
        block_print()
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.setParam("MIPGapAbs", self.BS_tol)
        env.setParam("Threads", self.solver_cores)
        m = gp.Model(env=env)
        enable_print()
        dummy = m.addVar(vtype=GRB.CONTINUOUS, name="dummy", lb=-GRB.INFINITY)
        pi = m.addVars(range(self.A), vtype=GRB.CONTINUOUS, name="pi")

        m.addConstr(gp.quicksum(pi) == 1)

        reward = []
        for a in range(self.A):
            reward_ = []
            for theta in AS:
                reward_.append(
                    sum(
                        [
                            self.probs[s, a][AS.index(theta), s_]
                            * (self.rewards[s, a, s_] + self.discount * v[s_])
                            for s_ in range(self.S)
                        ]
                    )
                )
                if tt + time.perf_counter() - start > self.timeout:
                    del env
                    del m
                    return ["T.O."]
            reward.append(reward_)

        reward = np.array(reward)
        m.addConstrs(
            dummy <= gp.quicksum(pi[a] * reward[a, i] for a in range(self.A))
            for i in range(len(AS))
        )

        dv = pi

        m.setObjective(dummy, GRB.MAXIMIZE)
        t_build = time.perf_counter() - start
        if tt + t_build > self.timeout:
            del env
            del m
            return ["T.O."]

        m.Params.TimeLimit = self.timeout - tt - t_build
        m.optimize()

        if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
            if method == "CS":
                del env
                return m, dv, dummy, reward
            if method == "LP":
                res = [m.objVal, np.array(m.getAttr("x", pi).values())]
                del m
                del env
                return res
        elif m.Status in [3, 4, 5]:
            del env
            del m
            return ["inf_unbd"]

    def projection_obj(self, theta, s, b, a):
        if self.dist == "binomial":
            return (self.N * (self.S - 1) * (theta - self.MLE[s, a]) ** 2) / (
                self.MLE[s, a] * (1 - self.MLE[s, a])
            )
        elif self.dist == "poisson":
            return (self.N * (theta - self.MLE[s, a]) ** 2) / self.MLE[s, a]

    def projection_deriv(self, theta, s, a):
        if self.dist == "binomial":
            return (2 * self.N * (self.S - 1) * (theta - self.MLE[s, a])) / (
                self.MLE[s, a] * (1 - self.MLE[s, a])
            )
        elif self.dist == "poisson":
            return (2 * self.N * (theta - self.MLE[s, a])) / (self.MLE[s, a])

    def solve_projection(self, s, b, a, beta, delta, tt, root_gap=0.01):
        start = time.perf_counter()
        left = self.timeout - tt
        if sum(self.P_hat[s, a] * b) <= beta:
            return [self.MLE[s, a], [0, 0]]
        # find root intervals
        if min(b) > beta:
            return [False, False]
        if self.dist == "binomial":
            theta_min = 0
            theta_max = 1
        elif self.dist == "poisson":
            theta_min = 0
            theta_max = self.MLE[s, a] + np.sqrt(
                (self.MLE[s, a] * chi2.ppf(1 - self.alpha, self.A) / self.N)
            )
        root_gap = max((theta_max - theta_min) / 100, 0.01)
        intervals = []
        for run in range(2):
            theta = self.MLE[s, a]
            EV = beta
            if time.perf_counter() - start > left:
                return ["T.O."]
            while EV >= beta:
                EV_old = EV
                if run == 0:
                    theta = max(theta - root_gap, theta_min)
                else:
                    theta = min(theta + root_gap, theta_max)
                if self.dist == "binomial":
                    EV = EV_jit(
                        np.array(binom_probs_fast(self.S, self.A, s, a, theta)), b
                    )
                elif self.dist == "poisson":
                    EV = EV_jit(
                        np.array(poisson_probs_fast(self.S, self.A, s, a, theta)), b
                    )
                # print(EV, beta, np.array(poisson_probs_fast(self.S, self.A, s, a, theta)))

                if theta == theta_min or theta == theta_max:
                    break
            if EV < beta:
                if run == 0:
                    intervals.append((run, theta, theta + root_gap))
                else:
                    intervals.append((run, theta - root_gap, theta))
            if run == 0 and len(intervals) == 1:
                inter = intervals[0]
                l, u = inter[1], inter[2]
                theta_max = min(self.MLE[s, a] + (self.MLE[s, a] - l), theta_max)
        if intervals == []:
            return [False, False]
        roots = []
        for run in range(len(intervals)):
            condition = True
            r, l, u = intervals[run]
            while condition:
                if time.perf_counter() - start > left:
                    return ["T.O."]
                x = (l + u) / 2
                objs = [self.projection_obj(sol, s, b, a) for sol in [l, u]]
                if max(objs) - min(objs) <= delta:
                    condition = False
                    continue
                if self.dist == "binomial":
                    P = np.array(binom_probs_fast(self.S, self.A, s, a, x))
                elif self.dist == "poisson":
                    P = np.array(poisson_probs_fast(self.S, self.A, s, a, x))
                if r == 0:
                    if EV_jit(P, b) <= beta:
                        l = x
                    else:
                        u = x
                else:
                    if EV_jit(P, b) <= beta:
                        u = x
                    else:
                        l = x
            roots.append((l, u))
        objs = [self.projection_obj((u + l) / 2, s, b, a) for (l, u) in roots]
        l, u = roots[np.argmin(objs)]
        opt = (l + u) / 2

        if l <= self.MLE[s, a]:
            objs_opt = [self.projection_obj(sol, s, b, a) for sol in [l, (u + l) / 2]]
        else:
            objs_opt = [self.projection_obj(sol, s, b, a) for sol in [u, (u + l) / 2]]

        return opt, np.array([min(objs_opt), max(objs_opt)])

    def Bellman_BS(self, v, s, t, tt):
        # print("---- Bellman update step t = %s for s = %s----\n" %(t,s))
        start = time.perf_counter()
        R_bar = np.max(self.rewards) / (1 - self.discount)
        delta = self.BS_tol * self.kappa / (2 * self.A * R_bar + self.A * self.BS_tol)
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
        condition = True
        i = 0
        while condition:
            conf = np.zeros((self.A, 2))
            if time.perf_counter() - start + tt > self.timeout:
                return ["T.O."]
            # print("l = ", l, ", u = ", u, "\n")
            x = (l + u) / 2
            theta_BS = np.zeros(self.A)
            for a in range(self.A):
                if sum(conf[:, 0]) > self.kappa:
                    break
                b = np.array(self.rewards[s, a]) + self.discount * np.array(v)
                # print("b = ", b, "theta = ", theta)
                sol = self.solve_projection(s, b, a, x, delta, tt)
                if sol[0] == "T.O.":
                    return ["T.O."]
                elif sol[0] == False:
                    theta_BS[a] = 0
                    conf[a] = np.array([self.kappa + 1, self.kappa + 1])
                else:
                    theta_BS[a], conf[a] = sol
            # print(d_bar)

            if sum(conf[:, 1]) <= self.kappa:
                u = x
            elif sum(conf[:, 0]) > self.kappa:
                l = x

            if (u - l <= self.BS_tol) or (
                self.kappa < sum(conf[:, 1]) and self.kappa >= sum(conf[:, 0])
            ):
                condition = False
            i += 1
        return x, theta_BS

    def Bellman_CS(self, v, s, t, tt, CS_tol=1e-6, k_max=100):
        start = time.perf_counter()
        theta_0 = self.Theta[s][0]
        Theta_k = [tuple(theta_0)]
        theta_k = Theta_k[0]
        Theta = [tuple(theta) for theta in self.Theta[s]]
        k = 0
        while k < k_max:
            tt_CS = time.perf_counter() - start
            if tt + tt_CS > self.timeout:
                try:
                    del m
                except NameError:
                    pass
                return ["T.O."]
            if k == 0:
                sol = self.Bellman_LP(v, s, t, Theta_k, tt, method="CS")
                if sol[0] == "T.O.":
                    try:
                        del m
                    except NameError:
                        pass
                    return ["T.O."]
                if sol[0] == "inf_unbd":
                    try:
                        del m
                    except NameError:
                        pass
                    return ["inf_unbd"]
                m, dv, dummy = sol[:3]
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
        del m
        return obj_k, pi_k

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
        start = time.perf_counter()
        t = 0
        v_new = np.zeros(self.S)
        pi_star = np.zeros((self.S, self.A))
        Delta = 10000
        worst_param = np.zeros((self.S, self.A))
        worst_dist = np.zeros((self.S, self.A, self.S))
        reasons = []
        theta_BS = np.zeros((self.S, self.A))
        s_VI = time.perf_counter()
        while (
            Delta >= self.VI_tol * (1 - self.discount) / (2 * self.discount)
        ) and t < self.t_max:
            # print("t=%s\n" %t)
            tt = time.perf_counter() - start
            if tt > self.timeout:
                return [v_new, t, tt]
            if t == 0:
                Delta = 0
            v = v_new.copy()
            for s in range(self.S):
                AS = [tuple(theta) for theta in self.Theta[s]]
                if method == "LP":
                    sol = self.Bellman_LP(v, s, t, AS, tt)
                    if sol[0] == "T.O.":
                        tt = time.perf_counter() - start
                        return [v_new, t, tt]  # timeout
                    elif sol[0] == "inf_unbd":
                        return ["inf_unbd"]
                    obj, pi_star[s] = sol

                elif method == "CS":
                    sol = self.Bellman_CS(v, s, t, tt)
                    if sol[0] == "T.O.":
                        tt = time.perf_counter() - start
                        return [v_new, t, tt]  # timeout

                    elif sol[0] == "inf_unbd":
                        return ["inf_unbd"]
                    obj, pi_star[s] = sol

                elif method == "PBS":
                    sol = self.Bellman_BS(v, s, t, tt)
                    if sol[0] == "T.O.":
                        tt = time.perf_counter() - start
                        return [v_new, t, tt]  # timeout
                    obj, theta = sol
                    theta_BS[s] = theta

                v_new[s] = obj

                Delta = max(np.array(v_new) - np.array(v))
            # print("ITER: ", t, "VALUES: ", v_new, "Delta: ", Delta,
            #     "Stopping: ", self.VI_tol * (1 - self.discount) / (2 * self.discount))
            t += 1
        t_VI = time.perf_counter() - start
        if method == "PBS":
            for s in range(self.S):
                sol = self.Bellman_CS(v_new, s, t, tt)
                if sol[0] == "T.O.":
                    tt = time.perf_counter() - start
                    return [v_new, t, tt]  # timeout
                obj, pi_star[s] = sol

        for s in range(self.S):
            reward = np.array(
                [
                    [
                        sum(
                            [
                                self.probs[s, a][i, s_]
                                * (self.rewards[s, a, s_] + self.discount * v_new[s_])
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

        return [
            np.round(pi_star, 4),
            np.round(v_new, 4),
            np.round(obj, 4),
            np.round(worst_param, 4),
            np.round(worst_dist, 4),
            t,
            np.round(t_VI, 4),
        ]
