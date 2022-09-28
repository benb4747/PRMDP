from numba import njit
from math import log, exp
import sys, os
import numpy as np
import pandas as pd

from scipy.stats import poisson


def read_results(file):
    file1 = open(file, "r")
    lines = file1.readlines()
    file1.close()

    lines_new = []
    names = eval(lines[0])
    for line in lines[1:]:
        line = line.rstrip("\n")
        if "failed" not in line:
            line = eval(line)
            for alg in ["CS", "LP", "PBS", "QP", "NBS"]:
                if line[names.index("t_VI_%s" % alg)] > line[names.index("timeout")]:
                    line[names.index("t_VI_%s" % alg)] = line[names.index("timeout")]
                    line[names.index("TO_%s" % alg)] = True
                if line[names.index("t_%s" % alg)] > line[names.index("timeout")]:
                    line[names.index("t_%s" % alg)] = line[names.index("timeout")]
                    line[names.index("TO_%s" % alg)] = True
            lines_new.append(line)
    print("Dataset has %s rows." % len(lines_new))

    res_array = np.array(lines_new)
    df = pd.DataFrame(data=res_array, columns=names, index=res_array[:, 0])
    return df


# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


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


@njit
def compute_conf_set(base, A, C, N, MLE, kappa):
    conf_set = []
    n = base.shape[0]
    for i in range(n):
        theta = base[i]
        if min(theta) >= 0 and max(theta) <= 1:
            conf = binom_conf_val(A, C, N, theta, MLE)
            if conf <= kappa:
                conf_set.append(theta)
    return conf_set


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


file1 = open("factorial.txt", "r")
fac = tuple(eval(file1.readlines()[0]))
file1.close()


@njit
def poisson_pmf(x, lam, fac=fac):
    if x < 0:
        return 0
    return (exp(-lam) * lam**x) / fac[x]


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


@njit
def poisson_conf_val(A, N, theta, MLE):
    val = 0
    for a in range(A):
        val += N * ((theta[a] - MLE[a]) ** 2) / (MLE[a])
    return val


@njit
def EV_jit(P, b):
    EV = 0
    for i in range(len(P)):
        EV += P[i] * b[i]
    return EV
