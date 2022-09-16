import sys, os
from src.NP_RMDP import NP_RMDP
from src.NV_RMDP import NV_RMDP
from src.P_RMDP import P_RMDP
from src.njit_functions import binompmf

import numpy as np
import itertools as it
import time
import datetime
import logging
from multiprocessing import Pool
from scipy.stats import chi2, binom
import pandas as pd


def test_algorithms(inp):
    (
        ind,
        q,
        h,
        c,
        b,
        C,
        discount,
        dist,
        t_max,
        eps,
        alpha,
        gap,
        N,
        timeout,
        solver_cores,
    ) = inp
    with open(start_file, "a") as myfile:
        myfile.write("Starting input %s.\n" % ind)
    headers = list(inp)
    S = C + 1
    A = C + 1
    P_0 = np.zeros(S)
    P_0[0] = 1
    p_0 = 0.75 * np.ones((S, A))
    MLE = np.zeros((S, A))
    _ = 0
    for (s, a) in it.product(range(S), range(A)):
        np.random.seed(ind * S * A + _)
        sample = binom.rvs(C, p_0[s, a], size=N)
        MLE[s, a] = sum(sample) / (N * C)
        if MLE[s, a] == 1:
            MLE[s, a] = 0.99
        elif MLE[s, a] == 0:
            MLE[s, a] = 0.01
        _ += 1

    P_hat = np.zeros((S, A, S), dtype="object")
    for (s, a, s_) in it.product(range(S), range(A), range(S)):
        if s_ == 0:
            P_hat[s, a, s_] = sum(
                [binompmf(C, MLE[s, a], d) for d in range(min(s + a, C), C + 1)]
            )
        else:
            P_hat[s, a, s_] = binompmf(C, MLE[s, a], min(s + a, C) - s_)
        P_hat[s, a, s_] = np.round(P_hat[s, a, s_], 6)

    # Parametric experiments
    P_NV = NV_RMDP(
        True,
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
        eps / 10,
        P_hat,
        "mchisq",
        chi2.ppf(1 - alpha, A),
        p_0,
        MLE,
        alpha,
        gap,
        N,
        timeout,
        solver_cores,
    )

    # Rewards, confidence set, PMFs
    P_NV.compute_rewards()
    P_NV_MDP = P_NV.construct_MDP()
    s = time.perf_counter()
    try:
        _ = P_NV_MDP.construct_conf()
    except MemoryError:
        with open(count_file, "a") as myfile:
            myfile.write(
                "Input %s had MemoryError while constructing base AS. \n" % ind
            )
        return
    e = time.perf_counter()
    if _ == np.array("T.O."):
        with open(count_file, "a") as myfile:
            myfile.write("Input %s timed out while constructing AS.\n" % ind)
            return
    t_AS = np.round(e - s, 3)
    left = timeout - t_AS
    headers.append(tuple(P_NV_MDP.num_params))
    # print("number of params: %s" %P_NV_MDP.num_params)
    s = time.perf_counter()
    _ = P_NV_MDP.compute_probs()
    e = time.perf_counter()
    if _ == np.array("T.O."):
        with open(count_file, "a") as myfile:
            myfile.write("Input %s timed out while computing probs.\n" % ind)
            return
    t_probs = np.round(e - s, 3)
    # print("Constructed AS, and computed probs in %s secs. \n" %(t_probs + t_AS))
    # Solve
    ## CS
    s = time.perf_counter()
    res_CS = P_NV_MDP.value_iteration(method="CS")
    e = time.perf_counter()
    if len(res_CS) == 3:
        v_CS, its_CS, t_VI_CS = res_CS
        v_CS = tuple(v_CS.flatten())
        pi_CS, obj_CS, theta_CS, P_CS = 4 * [-1]
        TO_CS = True
    elif res_CS[0] == "inf_unbd":
        with open(count_file, "a") as myfile:
            myfile.write("Input %s had an unbounded/inf model.\n" % ind)
            return
    else:
        pi_CS, v_CS, obj_CS, theta_CS, P_CS, its_CS, t_VI_CS = res_CS
        pi_CS, v_CS, P_CS, theta_CS = [
            tuple(pi_CS.flatten()),
            tuple(v_CS.flatten()),
            tuple(P_CS.flatten()),
            tuple(theta_CS.flatten()),
        ]
        TO_CS = False
    t_CS = np.round(e - s, 3)
    # print("solved with CS in %s seconds \n" %t_CS)

    ## BS
    s = time.perf_counter()
    res_BS = P_NV_MDP.value_iteration(method="BS")
    e = time.perf_counter()
    if len(res_BS) == 3:
        v_BS, its_BS, t_VI_BS = res_BS
        v_BS = tuple(v_BS.flatten())
        pi_BS, obj_BS, theta_BS, P_BS = 4 * [-1]
        TO_BS = True
    else:
        pi_BS, v_BS, obj_BS, theta_BS, P_BS, its_BS, t_VI_BS = res_BS
        pi_BS, v_BS, P_BS, theta_BS = [
            tuple(pi_BS.flatten()),
            tuple(v_BS.flatten()),
            tuple(P_BS.flatten()),
            tuple(theta_BS.flatten()),
        ]
        TO_BS = False
    t_BS = np.round(e - s, 3)
    # print("solved with BS in %s seconds \n" %t_BS)

    ## LP
    s = time.perf_counter()
    res_LP = P_NV_MDP.value_iteration(method="LP")
    e = time.perf_counter()
    if len(res_LP) == 3:
        v_LP, its_LP, t_VI_LP = res_LP
        v_LP = tuple(v_LP.flatten())
        pi_LP, obj_LP, theta_LP, P_LP = 4 * [-1]
        TO_LP = True
    elif res_LP[0] == ["inf_unbd"]:
        with open(count_file, "a") as myfile:
            myfile.write("Input %s had an unbounded/inf model.\n" % ind)
            return
    else:
        pi_LP, v_LP, obj_LP, theta_LP, P_LP, its_LP, t_VI_LP = res_LP
        pi_LP, v_LP, P_LP, theta_LP = [
            tuple(pi_LP.flatten()),
            tuple(v_LP.flatten()),
            tuple(P_LP.flatten()),
            tuple(theta_LP.flatten()),
        ]
        TO_LP = False
    t_LP = np.round(e - s, 3)
    # print("solved with LP in %s seconds \n" %t_LP)

    # Non parametric MDP
    NV = NV_RMDP(
        False,
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
        eps / 10,
        P_hat,
        "mchisq",
        chi2.ppf(1 - alpha, A) / N,
        p_0,
        MLE,
        alpha,
        gap,
        N,
        timeout,
        solver_cores,
    )

    NV.compute_rewards()
    NV_MDP = NV.construct_MDP()

    # Solve
    """## Projection algorithm where proj is solved as a QP
    s = time.perf_counter()
    res_proj_QP = NV_MDP.value_iteration(method="proj_qp")
    e = time.perf_counter()
    if type(res_proj_QP) == int:
        its_proj_QP = res_proj_QP
        pi_proj_QP, v_proj_QP, obj_proj_QP, P_proj_QP = 4 * [-1]
        TO_proj_QP = True
    elif len(res_proj_QP) == 2:
        v_proj_QP, its_proj_QP = res_proj_QP
        pi_proj_QP, obj_proj_QP, P_proj_QP = 3 * [-1]
        TO_proj_QP = True
    else:
        (
            pi_proj_QP,
            v_proj_QP,
            obj_proj_QP,
            P_proj_QP,
            its_proj_QP,
        ) = res_proj_QP
        TO_proj_QP = False
    t_proj_QP = np.round(e - s, 3)"""
    # print("solved with proj_QP in %s seconds \n" %t_proj_QP)

    ## Projection solved by sort algorithm
    s = time.perf_counter()
    res_proj_sort = NV_MDP.value_iteration(method="proj_sort")
    e = time.perf_counter()
    if len(res_proj_sort) == 3:
        v_proj_sort, its_proj_sort, t_VI_proj_sort = res_proj_sort
        v_proj_sort = tuple(v_proj_sort.flatten())
        pi_proj_sort, obj_proj_sort, P_proj_sort = 3 * [-1]
        TO_proj_sort = True
    else:
        (
            pi_proj_sort,
            v_proj_sort,
            obj_proj_sort,
            P_proj_sort,
            its_proj_sort,
            t_VI_proj_sort,
        ) = res_proj_sort
        pi_proj_sort, v_proj_sort, P_proj_sort = (
            tuple(pi_proj_sort.flatten()),
            tuple(v_proj_sort.flatten()),
            tuple(P_proj_sort.flatten()),
        )
        TO_proj_sort = False
    t_proj_sort = np.round(e - s, 3)
    # print("solved with proj_sort in %s seconds \n" %t_proj_sort)

    s = time.perf_counter()
    res_QP = NV_MDP.value_iteration(method="QP")
    e = time.perf_counter()
    if len(res_QP) == 4:
        v_QP, its_QP, t_VI_QP, reas_QP = res_QP
        v_QP = tuple(v_QP.flatten())
        pi_QP, obj_QP, P_QP = 3 * [-1]
        TO_QP = True
    else:
        pi_QP, v_QP, obj_QP, P_QP, its_QP, t_VI_QP = res_QP
        pi_QP, v_QP, P_QP = (
            tuple(pi_QP.flatten()),
            tuple(v_QP.flatten()),
            tuple(P_QP.flatten()),
        )
        reas_QP = -1
        TO_QP = False
    t_QP = np.round(e - s, 3)
    # print("solved with QP in %s seconds \n" %t_QP)

    res_list = headers + [
        t_AS,
        t_probs,
        pi_BS,
        v_BS,
        obj_BS,
        theta_BS,
        P_BS,
        its_BS,
        t_BS,
        t_VI_BS,
        TO_BS,
        pi_CS,
        v_CS,
        obj_CS,
        theta_CS,
        P_CS,
        its_CS,
        t_CS,
        t_VI_CS,
        TO_CS,
        pi_LP,
        v_LP,
        obj_LP,
        theta_LP,
        P_LP,
        its_LP,
        t_LP,
        t_VI_LP,
        TO_LP,
        pi_QP,
        v_QP,
        obj_QP,
        P_QP,
        its_QP,
        t_QP,
        t_VI_QP,
        TO_QP,
        reas_QP,
        pi_proj_sort,
        v_proj_sort,
        obj_proj_sort,
        P_proj_sort,
        its_proj_sort,
        t_proj_sort,
        t_VI_proj_sort,
        TO_proj_sort
        # tuple(pi_proj_QP.flatten()),
        # tuple(v_proj_QP),
        # obj_proj_QP,
        # tuple(P_proj_QP.flatten()),
        # its_proj_QP,
        # t_proj_QP,
        # TO_proj_QP
    ]

    with open(count_file, "a") as myfile:
        myfile.write("Finished solving input %s\n" % ind)

    with open(results_file, "a") as res_file:
        res_file.write(str(res_list) + "\n")

    return


def test_algorithms_mp(inp):
    ind = inp[0]
    try:
        return test_algorithms(inp)
    except Exception:
        with open(count_file, "a") as res_file:
            res_file.write("Input %s failed.\n" % ind)
        logging.exception("Input %s failed.\n" % ind)


cores = 32
loop_cores = 8
solver_cores = int(cores / loop_cores)

q_vals = [1, 5, 10]
h_vals = [1, 5, 10]
c_vals = [1, 5, 10]
b_vals = [1, 5, 10]

C_vals = [1, 2, 3, 7, 9, 14]  # inventory capacity, to ensure finite state space
discount = 0.5

dist = "binomial"
t_max = 1000
eps = 1e-6

alpha = 0.05
gap_vals = [0.20, 0.10, 0.05]
N_vals = [10, 50]

timeout = 4 * 60 * 60
inputs = [
    (q, h, c, b, C, discount, dist, t_max, eps, alpha, gap, N, timeout, solver_cores)
    for q in q_vals
    for h in h_vals
    for c in c_vals
    for b in b_vals
    for gap in gap_vals
    for N in N_vals
    for C in C_vals
    if q > c
]

for i in inputs:
    inputs[inputs.index(i)] = tuple([inputs.index(i)] + list(i))


names = [
    "ind",
    "q",
    "h",
    "c",
    "b",
    "C",
    "discount",
    "dist",
    "t_max",
    "eps",
    "alpha",
    "gap",
    "N",
    "timeout",
    "solver_cores",
    "num_params",
] + [
    "t_AS",
    "t_probs",
    "pi_BS",
    "v_BS",
    "obj_BS",
    "theta_BS",
    "P_BS",
    "its_BS",
    "t_BS",
    "t_VI_BS",
    "TO_BS",
    "pi_CS",
    "v_CS",
    "obj_CS",
    "theta_CS",
    "P_CS",
    "its_CS",
    "t_CS",
    "t_VI_CS",
    "TO_CS",
    "pi_LP",
    "v_LP",
    "obj_LP",
    "theta_LP",
    "P_LP",
    "its_LP",
    "t_LP",
    "t_VI_LP",
    "TO_LP",
    "pi_QP",
    "v_QP",
    "obj_QP",
    "P_QP",
    "its_QP",
    "t_QP",
    "t_VI_QP",
    "TO_QP",
    "reas_QP",
    "pi_proj_sort",
    "v_proj_sort",
    "obj_proj_sort",
    "P_proj_sort",
    "its_proj_sort",
    "t_proj_sort",
    "t_VI_proj_sort",
    "TO_proj_sort"
    # "pi_proj_QP",
    # "v_proj_QP",
    # "obj_proj_QP",
    # "P_proj_QP",
    # "its_proj_QP",
    # "t_proj_QP",
    # "TO_proj_QP",
]

test_full = inputs
h_ind = int(sys.argv[1]) - 1
start_file = "start_inf_NV.txt"
results_file = "results_inf_NV.txt"
count_file = "count_inf_NV.txt"

continuing = True

if continuing:
    file1 = open(results_file, "r")
    lines = file1.readlines()
    file1.close()

    done_ind = []
    failed = []
    # names = eval(lines[0])
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "failed" not in line:
            done_ind.append(eval(line)[0])

    file1 = open(count_file, "r")
    lines = file1.readlines()
    file1.close()

    timed_out = []
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "timed" in line:
            i = [int(i) for i in line.split() if i.isdigit()][0]
            timed_out.append(i)

    done = done_ind + failed + timed_out

    not_done = [i for i in test_full if i[0] not in done]

    test = not_done
else:
    test = test_full

# if T == 2 and continuing:
if h_ind == 0 and continuing:
    with open(count_file, "a") as myfile:
        myfile.write(
            "About to start solving the %s instances that didn't finish solving before. \n"
            % len(test)
        )

# wipes results file
if h_ind == 0 and not continuing:
    open(count_file, "w").close()
    open(results_file, "w").close()
    with open(count_file, "a") as myfile:
        myfile.write("About to start solving %s RMDP instances. \n" % len(test))
    with open(results_file, "a") as myfile:
        myfile.write(str(names) + "\n")

test = [i for i in test if i[2] == h_vals[h_ind]]
# test = [i for i in test if i[0] == 242]

if __name__ == "__main__":
    with Pool(processes=loop_cores) as p:
        res = list(p.imap(test_algorithms_mp, test))
