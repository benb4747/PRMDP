import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, poisson, binom
import gurobipy as gp
from gurobipy import GRB
import time
import itertools as it

from math import log, exp
from scipy.misc import derivative
from .njit_functions import block_print, enable_print


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
        timeout,
        solver_cores,
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
        self.timeout = timeout
        self.solver_cores = solver_cores

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

    def solve_projection(self, s, b, a, beta, delta, tt, method="sp"):
        start = time.perf_counter()
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
                if tt + (time.perf_counter() - start) > self.timeout:
                    return ["T.O."]
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
                block_print()
                env = gp.Env()
                env.setParam("OutputFlag", 0)
                env.setParam("Presolve", 1)
                env.setParam("LogToConsole", 0)
                env.setParam("Threads", self.solver_cores)
                m = gp.Model(env=env)
                enable_print()

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
                if tt + (time.perf_counter() - start) > self.timeout:
                    return ["T.O."]
                m.Params.TimeLimit = self.timeout - (tt + time.perf_counter() - start)
                m.setObjective(Obj, GRB.MAXIMIZE)

                m.optimize()
                if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
                    alpha_star = alpha.x
                    zeta_star = zeta.x

                    del m

                    return [
                        self.projection_obj_true(alpha_star, s, b, a, beta, zeta_star),
                        self.projection_obj_true(alpha_star, s, b, a, beta, zeta_star),
                    ]
                else:
                    return ["T.O."]

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
                        if tt + time.perf_counter() - start > self.timeout:
                            return ["T.O."]
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

    def Bellman_update(self, v, s, t, method, tt):
        # print("---- Bellman update step t = %s for s = %s----\n" %(t,s))
        start = time.perf_counter()
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
                if tt + time.perf_counter() - start > self.timeout:
                    return ["T.O."]
                # print("l = ", l, ", u = ", u, "\n")
                theta = (l + u) / 2
                d_bar = np.zeros((self.A, 2))
                for a in range(self.A):
                    b = np.array(self.rewards[s, a]) + self.discount * np.array(v)
                    # print("b = ", b, "theta = ", theta)
                    d_bar[a] = self.solve_projection(s, b, a, theta, delta, tt, method)
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
            sol = self.find_policy(s, v, tt)
            if len(sol) == 1:
                return ["T.O."]
            obj = sol[-1]
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
        start = time.perf_counter()
        tt = 0
        t = 0
        v_new = np.zeros(self.S)
        pi_star = np.zeros((self.S, self.A))
        Delta = 10000
        alpha = np.zeros((self.S, self.A))
        worst = np.zeros((self.S, self.A, self.S))
        while (
            Delta >= self.eps * (1 - self.discount) / (2 * self.discount)
        ) and t < self.t_max:
            tt += time.perf_counter() - start
            if t == 0:
                Delta = 0
            v = v_new.copy()
            for s in range(self.S):
                sol = self.Bellman_update(v, s, t, method, tt)
                if len(sol) == 1:
                    return ["T.O."]
                v_new[s], alpha[s] = sol
            Delta = max(np.array(v_new) - np.array(v))
            t += 1
            # print("Iteration %s values: %s \n" %(t, v_new))
        self.values = v_new

        for s in range(self.S):
            sol = self.find_policy(s, v_new, tt)
            if len(sol) == 1:
                return [v_new, t]
            pi_star[s], worst[s] = sol[:2]

        obj = np.matmul(np.array(self.P_0), np.array(v_new))

        return {
            "Policy": np.round(pi_star, 4),
            "Values": np.round(v_new, 4),
            "Objective": np.round(obj, 4),
            "Worst": np.round(worst, 4),
            "Nits": t,
        }

    def find_policy(self, s, v, tt):
        start = time.perf_counter()
        block_print()
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        env.setParam("Presolve", 1)
        env.setParam("LogToConsole", 0)
        env.setParam("Threads", self.solver_cores)
        m = gp.Model(env=env)
        enable_print()
        # print("About to start building model for the policy. \n")
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

        if tt + time.perf_counter() - start > self.timeout:
            return ["T.O."]

        m.Params.TimeLimit = self.timeout - (tt + time.perf_counter() - start)
        m.setObjective(Obj, GRB.MAXIMIZE)
        # print("About to start solving for the policy. \n")
        m.optimize()

        if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
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
        else:
            del m
            return ["T.O."]
