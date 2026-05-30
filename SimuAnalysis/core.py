import os
import sys
from copy import deepcopy

import numpy as np
import scipy.stats as stats

import config


MAIN_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "Main")
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)

from tools import *  # noqa: E402,F403
from Agents import *  # noqa: E402,F403
from Predictor import *  # noqa: E402,F403
from GLCP import GLCP, SCC  # noqa: E402
from SLCP import SLCP, SLCP_SCC  # noqa: E402
from SSAE import dissemiGLCP, dissemiSCC  # noqa: E402
from QuanRegressor import QRModel  # noqa: E402
from engGenerator import Generator  # noqa: E402


SUPPORTED_SIGMA_DTYPES = [
    "abs",
    "exp",
    "quad",
    "logabs",
    "relu",
    "softplus",
    "sigmoid",
    "sqrtabs",
    "maxabs",
    "l1",
    "l2",
    "sin",
    "mix",
]


def default_lbd_grid():
    return list(config.LBDS)


def _ensure_2d(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((1, -1))
    return arr


def sigma(X, gamma, dtype="abs"):
    X = _ensure_2d(X)
    abs_x = np.abs(X)
    d = X.shape[1]
    root_d = np.sqrt(d)
    clipped = np.clip(X, -20.0, 20.0)

    if dtype == "abs":
        base = np.abs((abs_x - np.sqrt(2.0 / np.pi)).sum(-1)) / root_d
    elif dtype == "exp":
        base = np.exp(np.minimum(abs_x, 3.0)).sum(-1) / root_d
    elif dtype == "quad":
        base = (X ** 2).sum(-1) / root_d
    elif dtype == "logabs":
        base = np.log1p(abs_x).sum(-1) / root_d
    elif dtype == "relu":
        base = np.maximum(X, 0.0).sum(-1) / root_d + 0.1
    elif dtype == "softplus":
        base = np.log1p(np.exp(clipped)).sum(-1) / root_d
    elif dtype == "sigmoid":
        base = (1.0 / (1.0 + np.exp(-clipped))).sum(-1) / root_d
    elif dtype == "sqrtabs":
        base = np.sqrt(abs_x + 0.25).sum(-1) / root_d
    elif dtype == "maxabs":
        base = np.max(abs_x, axis=-1) + 0.1
    elif dtype == "l1":
        base = abs_x.sum(-1) / d + 0.1
    elif dtype == "l2":
        base = np.sqrt((X ** 2).sum(-1)) / root_d + 0.1
    elif dtype == "sin":
        base = (1.25 + np.sin(X)).sum(-1) / root_d
    elif dtype == "mix":
        base = (0.5 * abs_x.sum(-1) + 0.5 * (X ** 2).sum(-1)) / root_d
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    base = np.maximum(np.asarray(base, dtype=float), 1e-8)
    return base.reshape((-1, 1)) * np.sqrt(gamma)


def generate_agent(n, d, me, gamma, mu, dtype="abs"):
    X = np.random.normal(0, 1, (n, d)) + mu.reshape((1, -1))
    noise = np.random.normal(0, 1, (n, 1)) * sigma(X, gamma, dtype)
    Y = X.sum(-1).reshape((-1, 1)) / me + noise
    return Agent(d, n, X, Y)


def true_coverage(interval, x, me, gamma, dtype="abs"):
    inter = [(num - x.sum() / me) / sigma(x, gamma, dtype=dtype).item() for num in interval]
    return stats.norm.cdf(inter[1]) - stats.norm.cdf(inter[0])


def summation_param(COV, SIZE, alpha=0.1):
    mar = np.mean(np.mean(COV, axis=-1), axis=-1)
    size = np.mean(np.mean(SIZE, axis=-1), axis=-1)
    size_std = np.std(np.mean(SIZE, axis=-1), axis=-1)
    local_mar = COV.mean(1)
    local_cov = np.mean(np.abs(local_mar - (1 - alpha)), axis=-1)
    return mar, size, size_std, local_cov


def _eval_cs(cs, testX, me, gamma_t, dtype, isinf=None):
    testN = testX.shape[0]
    cov = np.zeros(testN)
    size = cs[:, 1] - cs[:, 0]
    if isinf is None:
        isinf = np.zeros(testN, dtype=bool)
    for j in range(testN):
        c = true_coverage(cs[j, :], testX[j, :], me, gamma_t, dtype=dtype)
        cov[j] = 1.0 if isinf[j] else c
    return cov, size


def _unpack_prediction(pred_out):
    if isinstance(pred_out, tuple):
        return pred_out[0], pred_out[1]
    return pred_out, None


def _aggregate(cov, size, alpha=0.1):
    def _reduce_metrics(metrics):
        mar, _, size_m, size_std, local_cov, _ = metrics
        return mar, size_m, size_std, local_cov

    if cov.ndim == 2:
        return list(_reduce_metrics(summation(cov, size, alpha)))
    if cov.ndim == 3:
        stats_by_method = [_reduce_metrics(summation(cov[i], size[i], alpha)) for i in range(cov.shape[0])]
        return [np.array([s[k] for s in stats_by_method], dtype=float) for k in range(4)]
    raise ValueError(f"Unsupported cov ndim for _aggregate: {cov.ndim}")


def _aggregate_grid(cov, size, alpha=0.1):
    if cov.ndim == 3:
        return list(summation_param(cov, size, alpha))
    if cov.ndim == 4:
        stats_by_method = [summation_param(cov[i], size[i], alpha) for i in range(cov.shape[0])]
        return [np.array([s[k] for s in stats_by_method], dtype=float) for k in range(4)]
    raise ValueError(f"Unsupported cov ndim for _aggregate_grid: {cov.ndim}")


def run_experiment(
    d=10,
    r=0.5,
    n=30,
    m=500,
    N=2000,
    gamma_t=1.0,
    gamma_s=1.0,
    dtype="abs",
    alpha=0.1,
    repeats=50,
    testN=1000,
    hidden_dim=None,
    epoches=200,
    n_grid=50,
    lbds=None,
    temperature=10.0,
    alpha_tol=0.02,
    run_sdcp=True,
    run_ppi=True,
    run_noal=True,
    run_sel=True,
):
    if dtype not in SUPPORTED_SIGMA_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if hidden_dim is None:
        hidden_dim = [30, 50, 50, 30]
    if lbds is None:
        lbds = default_lbd_grid()
    lbds = sorted(set(float(x) for x in lbds))
    if 0.0 not in lbds:
        lbds = [0.0] + lbds

    mu_t = np.ones(d) / np.sqrt(d) * r
    mu_s = np.zeros(d)
    me_t, me_s = d / 2, d / 3

    method_num = 2
    L = len(lbds)
    cov_base, size_base = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))
    cov_orac, size_orac = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))
    cov_slcp, size_slcp = np.zeros((method_num, L, repeats, testN)), np.zeros((method_num, L, repeats, testN))
    cov_sel, size_sel = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))
    selected_idx = np.zeros((method_num, repeats), dtype=int)

    cov_sdcp = size_sdcp = cov_ppi = size_ppi = cov_noal = size_noal = None
    if run_sdcp:
        cov_sdcp, size_sdcp = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))
    if run_ppi:
        cov_ppi, size_ppi = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))
    if run_noal:
        cov_noal, size_noal = np.zeros((method_num, repeats, testN)), np.zeros((method_num, repeats, testN))

    setseed(repeats + 100)
    trAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    predAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    pred = Predictor("lr", fit_intercept=False)
    pred.trainFromAgent(trAgent)
    predAgent.calScore(pred, defaultScore)
    generator_base = Generator(d, hidden_dim, d)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, epoches, 5e-3, mute=True)

    for rep in range(repeats):
        seed_rep = 1 + rep
        setseed(seed_rep)

        testAgent = generate_agent(testN, d, me_t, gamma_t, mu_t, dtype)
        calTrAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        calAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        semiAgent = generate_agent(m, d, me_t, gamma_t, mu_t, dtype)
        semiX = semiAgent.X
        calOrac = generate_agent(2000, d, me_t, gamma_t, mu_t, dtype)

        setseed(seed_rep)
        predictor = Predictor("lr", fit_intercept=False)
        predictor.trainFromAgent(calTrAgent)
        calAgent.calScore(predictor, defaultScore)
        calTrAgent.calScore(predictor, defaultScore)
        calOrac.calScore(predictor, defaultScore)

        setseed(seed_rep)
        generator = deepcopy(generator_base)
        generator.cal_scalar(calTrAgent.getX(), calTrAgent.getS(), 200, stat_type="CvM")
        qrmodel = QRModel("CDF", 1 - alpha, CDFModel=generator, d=d)

        setseed(seed_rep)
        glcp = GLCP(calAgent, deepcopy(generator), predictor, 500, alpha)
        cs, isinf = glcp.predict(testAgent.getX(), defaultSolveScore)
        cov_base[0, rep], size_base[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        setseed(seed_rep)
        scc = SCC(calAgent, deepcopy(qrmodel), predictor, alpha)
        cs, isinf = scc.predict(testAgent.getX(), defaultSolveScore)
        cov_base[1, rep], size_base[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        setseed(seed_rep)
        slcp = SLCP(calAgent, semiX, deepcopy(generator), predictor, [2])
        targ_alpha = float(np.clip(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n, 1e-6, 1 - 1e-6))
        tuner_list = slcp.tune_lbd_list(
            5,
            epoches,
            5e-3,
            int(n_grid),
            lbds,
            temperature=temperature,
            tol_gap=0.001,
            max_iter=10000,
            m=200,
            targ_alpha=targ_alpha,
            penalty="MSE",
        )
        for i in range(L):
            slcp.load_tuner(tuner_list[i], m=200, n=5, temperature=temperature, alpha=alpha)
            cs, isinf = _unpack_prediction(slcp.predict(testAgent.getX(), defaultSolveScore))
            cov_slcp[0, i, rep], size_slcp[0, i, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        if run_sel:
            _, _, idx = slcp.auto_lbd_tune(
                5,
                epoches,
                5e-3,
                alpha,
                int(n_grid),
                lbds,
                temperature=temperature,
                gap=alpha_tol,
                tol_gap=0.001,
                max_iter=10000,
                m=200,
                targ_alpha=targ_alpha,
                penalty="MSE",
                tuner_list=tuner_list,
            )
            selected_idx[0, rep] = idx
            cs, isinf = _unpack_prediction(slcp.predict(testAgent.getX(), defaultSolveScore))
            cov_sel[0, rep], size_sel[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        setseed(seed_rep)
        slcp_scc = SLCP_SCC(calAgent, semiX, deepcopy(generator), deepcopy(qrmodel), predictor, [2])
        tuner_list_scc = slcp_scc.tune_lbd_list(
            5,
            epoches,
            5e-3,
            int(n_grid),
            lbds,
            tol_gap=0.002,
            max_iter=10000,
            targ_alpha=targ_alpha,
            penalty="MSE",
        )
        for i in range(L):
            slcp_scc.load_tuner(tuner_list_scc[i], n=5, alpha=alpha)
            cs, isinf = _unpack_prediction(slcp_scc.predict(testAgent.getX(), defaultSolveScore))
            cov_slcp[1, i, rep], size_slcp[1, i, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        if run_sel:
            _, _, idx = slcp_scc.auto_lbd_tune(
                5,
                epoches,
                5e-3,
                alpha,
                int(n_grid),
                lbds,
                gap=alpha_tol,
                tol_gap=0.002,
                max_iter=10000,
                targ_alpha=targ_alpha,
                penalty="MSE",
                tuner_list=tuner_list_scc,
            )
            selected_idx[1, rep] = idx
            cs, isinf = _unpack_prediction(slcp_scc.predict(testAgent.getX(), defaultSolveScore))
            cov_sel[1, rep], size_sel[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        setseed(seed_rep)
        glcp_orac = GLCP(calOrac, deepcopy(generator), predictor, 500, alpha)
        cs, isinf = glcp_orac.predict(testAgent.getX(), defaultSolveScore)
        cov_orac[0, rep], size_orac[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        setseed(seed_rep)
        scc_orac = SCC(calOrac, deepcopy(qrmodel), predictor, alpha)
        cs, isinf = scc_orac.predict(testAgent.getX(), defaultSolveScore)
        cov_orac[1, rep], size_orac[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        if run_sdcp:
            setseed(seed_rep)
            sdcp = dissemiGLCP(
                calAgent,
                deepcopy(generator),
                predictor,
                semiX,
                200,
                alpha,
                hiddenDim=hidden_dim,
                batch_size=32,
                epochs=epoches,
                learning_rate=5e-3,
                m=200,
                iftune=True,
            )
            cs, isinf = sdcp.predict(testAgent.getX(), defaultSolveScore)
            cov_sdcp[0, rep], size_sdcp[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

            sdcp_scc = dissemiSCC(
                calAgent,
                deepcopy(qrmodel),
                predictor,
                semiX,
                alpha,
                hiddenDim=hidden_dim,
                batch_size=32,
                epochs=epoches,
                learning_rate=5e-3,
                m=200,
                iftune=True,
            )
            cs, isinf = sdcp_scc.predict(testAgent.getX(), defaultSolveScore)
            cov_sdcp[1, rep], size_sdcp[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        if run_ppi:
            setseed(seed_rep)
            outer_beta = GLCP(predAgent, deepcopy(generator), predictor, 200, alpha).beta
            ppi = dissemiGLCP(
                calAgent,
                deepcopy(generator),
                predictor,
                semiX,
                200,
                alpha,
                outer=True,
                outerX=predAgent.getX(),
                outerbeta=outer_beta,
                hiddenDim=hidden_dim,
                batch_size=32,
                epochs=20,
                learning_rate=5e-3,
                m=200,
                iftune=True,
            )
            cs, isinf = ppi.predict(testAgent.getX(), defaultSolveScore)
            cov_ppi[0, rep], size_ppi[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

            outer_beta_scc = SCC(predAgent, deepcopy(qrmodel), predictor, alpha).beta
            ppi_scc = dissemiSCC(
                calAgent,
                deepcopy(qrmodel),
                predictor,
                semiX,
                alpha,
                outer=True,
                outerX=predAgent.getX(),
                outerbeta=outer_beta_scc,
                hiddenDim=hidden_dim,
                batch_size=32,
                epochs=20,
                learning_rate=5e-3,
                m=200,
                iftune=True,
            )
            cs, isinf = ppi_scc.predict(testAgent.getX(), defaultSolveScore)
            cov_ppi[1, rep], size_ppi[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

        if run_noal:
            setseed(seed_rep)
            load_q = np.quantile(
                generator.cdf(semiX, generator.generaten(semiX, 5), 200).reshape(-1),
                (1 - alpha) * (calAgent.n + 1) / calAgent.n,
                method="higher",
            )
            noal = GLCP(calAgent, deepcopy(generator), predictor, 200, alpha, loadq=True, q=load_q)
            cs, isinf = noal.predict(testAgent.getX(), defaultSolveScore)
            cov_noal[0, rep], size_noal[0, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

            load_qhat = np.quantile(
                (generator.generaten(semiX, 5) - qrmodel.predict(semiX)).reshape(-1),
                (1 - alpha) * (calAgent.n + 1) / calAgent.n,
                method="higher",
            )
            noal_scc = SCC(calAgent, deepcopy(qrmodel), predictor, alpha, loadq=True, qhat=load_qhat)
            cs, isinf = noal_scc.predict(testAgent.getX(), defaultSolveScore)
            cov_noal[1, rep], size_noal[1, rep] = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)

    res = {
        "meta": {
            "d": d,
            "r": r,
            "n": n,
            "m": m,
            "N": N,
            "gamma_t": gamma_t,
            "gamma_s": gamma_s,
            "dtype": dtype,
            "alpha": alpha,
            "repeats": repeats,
            "testN": testN,
            "n_grid": n_grid,
            "lbds": lbds,
            "temperature": temperature,
            "alpha_tol": alpha_tol,
            "method_order": ["GLCP", "SCC"],
            "sigma_family": dtype,
            "available_sigma_families": list(SUPPORTED_SIGMA_DTYPES),
        },
        "base": _aggregate(cov_base, size_base, alpha),
        "oracle": _aggregate(cov_orac, size_orac, alpha),
        "StCP": _aggregate_grid(cov_slcp, size_slcp, alpha),
    }

    if run_sel:
        res["StCP-sel"] = _aggregate(cov_sel, size_sel, alpha)
        res["selected_lambda_idx"] = selected_idx
        res["selected_lambda"] = np.array(lbds)[selected_idx]
    if run_sdcp:
        res["SDCP"] = _aggregate(cov_sdcp, size_sdcp, alpha)
    if run_ppi:
        res["PPI"] = _aggregate(cov_ppi, size_ppi, alpha)
    if run_noal:
        res["NOAL"] = _aggregate(cov_noal, size_noal, alpha)

    return res
