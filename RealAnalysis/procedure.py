import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
from sklearn.cluster import KMeans
from tools import *
from Agents import *
from Predictor import *
from GLCP import *
from SLCP import *
from SCP import *
from SSAE import *
from CNNnet import *
from PPI import *
import warnings
import datetime
from copy import deepcopy
import itertools
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

class SimSSGenerator:
    def __init__(self, ss_inputDim, ss_hiddenDim, eng_hiddenDim:list[int], eng_randNum:int=1, **kwargs):
        self.ae = AutoEncoder(ss_inputDim, ss_hiddenDim)
        self.generator = Generator(ss_hiddenDim[-1], eng_hiddenDim, eng_randNum, **kwargs)

    def train(self, X:np.ndarray, Y:np.ndarray, m:int=100, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, **kwargs):
        self.ae.train_loop(X, epochs=epochs, lr=learning_rate, batch_size=batch_size, verbose=False, **kwargs)
        self.generator.trainEng(self.dimdeduc(X, dtype='ndarray'), Y, m=m, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, **kwargs)

    def dimdeduc(self, X, dtype='tensor'):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.ae.input_dim)
        return self.ae.encoder(X).detach() if dtype == 'tensor' else self.ae.encoder(X).detach().numpy()

    def generaten(self, x: np.ndarray, n: int = 1, seed=0):
        x_deduc = self.dimdeduc(x, dtype='ndarray')
        return self.generator.generaten(x_deduc)

    def cal_scalar(self, X:np.ndarray, Y:np.ndarray, m:int=100, seed=0, stat_type='CvM'):
        return self.generator.cal_scalar(self.dimdeduc(X, dtype='ndarray'), Y, m=m, seed=seed, stat_type=stat_type)

    def cdf(self, X: np.ndarray, Y: np.ndarray, m: int = 100, **kwargs):
        return self.generator.cdf(self.dimdeduc(X, dtype='ndarray'), Y, m=m, **kwargs)

    def quantile(self, X: np.ndarray, q: float = .9, m: int = 100, **kwargs):
        return self.generator.quantile(self.dimdeduc(X, dtype='ndarray'), q=q, m=m, **kwargs)

def assertDir(p, isfile=False):
    if isfile:
        file_path = Path(p)
        path_obj = file_path.parent
    else:
        path_obj = Path(p)
    path_obj.mkdir(parents=True, exist_ok=True)

def auto_skew_transform(df, log, path, isLog, skew_threshold=1.0):
    skewed_features = []
    col_list = []
    skewness_list = []
    for col in df.columns:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_features.append(col)
            df[col] = np.log1p(df[col])
            col_list.append(col)
            skewness_list.append(skewness)
    log(f"Applied log1p to "+', '.join([f"{col_list[i]} ({skewness_list[i]:.2f})" for i in range(len(col_list))]), path, islog=isLog)
    return df, skewed_features

def summation_real(cov, size, IND, fullIND, alpha=.1):
    mar = np.mean(cov)
    mar_std = np.std(np.mean(cov, axis=1))
    mean_size = np.mean(size)
    size_std = np.std(np.mean(size, axis=1))
    miscov = np.zeros(fullIND.shape[1])
    for i in range(fullIND.shape[1]):
        idxs, counts = np.unique(fullIND[:, i], return_counts=True)
        loc_cov = np.zeros(len(idxs))
        weight = np.zeros(len(idxs))
        id_matx = fullIND[:, i][IND]
        for j in range(len(idxs)):
            idx = idxs[j]
            loc_cov[j] = np.mean(cov[id_matx == idx])
            weight[j] = np.sum(id_matx == idx)
        miscov[i] = np.sum(np.abs(loc_cov - (1-alpha)) * weight / weight.sum())
    return mar, mar_std, mean_size, size_std, miscov

def summation_real_cls(cov, size, IND, fullIND, alpha=.1):
    mar = np.mean(cov)
    mar_std = np.std(np.mean(cov, axis=1))
    mean_size = np.mean(size)
    size_std = np.std(np.mean(size, axis=1))
    idxs, counts = np.unique(fullIND, return_counts=True)
    loc_cov = np.zeros(len(idxs))
    weight = np.zeros(len(idxs))
    id_matx = fullIND[IND]
    for j in range(len(idxs)):
        idx = idxs[j]
        loc_cov[j] = np.mean(cov[id_matx == idx])
        weight[j] = np.sum(id_matx == idx)
    miscov = np.sum(np.abs(loc_cov - (1-alpha)) * weight / weight.sum())
    return mar, mar_std, mean_size, size_std, miscov

def logResult(mar, mar_std, size, size_std, local_cov, name, isLog=False, path='', log=log):
    miscov = ' '.join([f"{x:.4f}" for x in local_cov])
    log(f"{name.upper()} : mar : {mar:.4f}, size : {size:.4f}, mar std : {mar_std:.4f}, size std : {size_std:.4f}, miscoverage : ({miscov})", path=path, islog=isLog)

def logResultParam(mar, mar_std, size, size_std, local_cov, name, param_comb,  isLog=False, path='', log=log):
    for param_idx in range(len(param_comb)):
        n_grid, lbd, temperature = param_comb[param_idx]
        miscov = ' '.join([f"{x:.4f}" for x in local_cov[param_idx]])
        log(f"{name.upper()} ({n_grid}, {lbd}, {temperature}) : mar : {mar[param_idx]:.4f}, mean size : {size[param_idx]:.4f}, mar std : {mar_std[param_idx]:.4f}, size std : {size_std[param_idx]:.4f}, miscoverage : ({miscov})", path=path, islog=isLog)

def logResult_cls(mar, mar_std, size, size_std, local_cov, name, isLog=False, path='', log=log):
    log(f"{name.upper()} : mar : {mar:.4f}, size : {size:.4f}, mar std : {mar_std:.4f}, size std : {size_std:.4f}, miscoverage : ({local_cov:.4f})", path=path, islog=isLog)

def logResultParam_cls(mar, mar_std, size, size_std, local_cov, name, param_comb,  isLog=False, path='', log=log):
    for param_idx in range(len(param_comb)):
        n_grid, lbd, temperature = param_comb[param_idx]
        log(f"{name.upper()} ({n_grid}, {lbd}, {temperature}) : mar : {mar[param_idx]:.4f}, mean size : {size[param_idx]:.4f}, mar std : {mar_std[param_idx]:.4f}, size std : {size_std[param_idx]:.4f}, miscoverage : ({local_cov[param_idx]:.4f})", path=path, islog=isLog)

def logInfo(dic, Lpath):
    for k, v in dic.items():
        log(f"{k} : {v}", path=Lpath, islog=True)

def kernel_weight(y, loc, scale):
    w = np.exp(-(y-loc)**2/(2*scale**2))
    w = w / np.sum(w)
    return w

def selectN(y, loc, scale, n):
    w = kernel_weight(y, loc, scale)
    full_ind = np.array(range(len(y)))
    ind = np.random.choice(full_ind, size=n, replace=False, p=w)
    mask = np.zeros(len(y), dtype=bool)
    mask[ind] = True
    return mask

def splitData(X, y, mask):
    """
    根据ind指标划分出两组data
    :return:
    """
    X1, y1 = X[mask], y[mask]
    X2, y2 = X[~mask], y[~mask]
    return X1, y1, X2, y2

def splitX(X, k):
    KM = KMeans(n_clusters=k, random_state=42)
    KM.fit(X)
    return KM

def selectTarget(images, labels, tar_N, aux_N):
    images_tar = np.zeros([0]+list(images.shape[1:]))
    images_aux = np.zeros([0]+list(images.shape[1:]))
    labels_tar = np.zeros((0, 1))
    labels_aux = np.zeros((0, 1))
    for i in range(len(tar_N)):
        mask = (labels == i).squeeze()
        X_, Y_ = images[mask], labels[mask]
        inds = np.random.permutation(len(X_))
        images_tar = np.concatenate([images_tar, X_[inds[:tar_N[i]]]])
        labels_tar = np.concatenate([labels_tar, Y_[inds[:tar_N[i]]]])
        images_aux = np.concatenate([images_aux, X_[inds[tar_N[i]:]]])
        labels_aux = np.concatenate([labels_aux, Y_[inds[tar_N[i]:]]])
    agent_target = Agent(0, np.sum(tar_N), images_tar, labels_tar)
    agent_aux = Agent(0, np.sum(aux_N), images_aux, labels_aux)
    return agent_target, agent_aux

def combineImgAgents(agentList:list[Agent]):
    X = agentList[0].X
    Y = agentList[0].Y
    n = agentList[0].n
    for agent in agentList[1:]:
        X = np.concatenate([X, agent.X], axis=0)
        Y = np.concatenate([Y, agent.Y], axis=0)
        n += agent.n
    agent = Agent(0, n, X, Y)
    return agent

def Cl_solveScore(testX, sq, pred):
    if len(testX.shape) == 1:
        yhat = predictor.predict_feat(testX.reshape((1, -1)))
        return np.astype(yhat >= 1-sq, np.float32), False
    else:
        probs = pred.predict_feat(testX)
        return np.astype(probs >= 1-sq.reshape((-1,1)), np.float32), np.zeros(testX.shape[0], dtype=bool)

def sum_compare_result(resDict, name_, path='', isLog=False, tol=0.02, n=30, cls=False):
    res = np.zeros((7, 4))
    key_list = [f'{name_}', f'{name_} SDCP', f'{name_} PPI', f'{name_} SLCP', f'{name_} SLCP-sel', f'{name_} ORCP', f'{name_} NOAL']
    for idx in [0, 1, 2, 4, 5, 6]:
        values = resDict[key_list[idx]]
        res[idx, :] = np.array([values[0], values[2], values[3], values[4][0]]) if not cls else np.array([values[0], values[2], values[3], values[4]])
    tol_std = [res[0, 2]]
    for idx in [1, 2]:
        if 0.9-tol <= res[idx, 0] <= 0.9+1/(n+1):
            tol_std.append(res[idx, 2])
    tol_min_std = np.min(tol_std)
    for i in [3]:
        mar, _, size, std, mis = resDict[key_list[i]]
        mask = (std < tol_min_std) & (std >= res[-1, 2]) & (mar >= 0.9-tol) & (mar <= 0.9+1/(n+1))
        if np.sum(mask) == 0:
            mask = (mar >= 0.9-tol) & (mar <= 0.9+1/(n+1))
        if np.sum(mask) == 0:
            idx = np.argmin(np.abs(np.array(mar) - 0.9))
            res[i, :] = (mar[idx], size[idx], std[idx], mis[idx, 0]) if not cls else (mar[idx], size[idx], std[idx], mis[idx])
        else:
            mar, size, std, mis = np.array(mar)[mask], np.array(size)[mask], np.array(std)[mask], np.array(mis)[mask]
            idx = np.argmin(std)
            res[i, :] = (mar[idx], size[idx], std[idx], mis[idx, 0]) if not cls else (mar[idx], size[idx], std[idx], mis[idx])
    for i in range(7):
        log(f"{key_list[i]}: \t"+' '.join([f"{res[i, j]:.5f}" for j in range(4)]), path=path, islog=isLog)
    return res

class EngPredictor:
    def __init__(self, eng):
        self.model = eng
    def predict(self, X):
        return self.model.generaten(X)

def procedure_reg(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, hidden_dim, noise_dim, epoches, alpha, agent_target:Agent, agent_aux:Agent, tol_gap:float=2e-2, defaultScore=defaultScore, defaultSolveScore=defaultSolveScore, predmethod=None, seed=0, Lpath='', SumLpath='', SimRpath='', SimName='', isLog=False, addDict=None, comb_aux_tr=False, **kwargs):
    """

    :param n_grids: SLCP 优化参数
    :param lbds: SLCP 优化参数
    :param temperatures: SLCP 优化参数
    :param repeats: 模拟次数
    :param testN: 测试样本量
    :param m: unlabelled数量
    :param n: 有标签calibration量 n//2
    :param N: source训练1、训练2样本量 N//3
    :param d: 数据维度
    :param hidden_dim: engressor训练参数
    :param noise_dim: engressor训练参数
    :param epoches: engressor训练参数
    :param alpha: level
    :param agent_target: target数据池
    :param agent_aux: source数据池
    :param defaultScore:
    :param defaultSolveScore:
    :param seed:
    :param Lpath:
    :param isLog:
    :return:
    """
    if isLog:
        assertDir(SimRpath, False)
        assertDir(Lpath, True)
        assertDir(SumLpath, True)
        relog(SumLpath)
        relog(Lpath)
        dic = {'labeled target data': n,
               'unlabeled target data': m,
               'test data': testN,
               'labeled source data': N,
               'seed': seed, 'alpha': alpha, 'repeats': repeats,
               'Engression training epoches': epoches,
               'Engression hidden_dim': hidden_dim,
               "Engression noise_dim": noise_dim,
               "Engression tuning quantile grid number": n_grids,
               "Engression tuning penalty coefficient lambda": lbds,
               "Engression tuning smoothing temperature": temperatures,
               } | addDict
        logInfo(dic, Lpath)
    X_test_group = [10, 15, 20, 25, 30]
    param_comb = list(itertools.product(n_grids, lbds, temperatures))
    CS1, COV1, SIZE1 = generate_EmptyK(repeats, testN, 2)
    CS2, COV2, SIZE2 = generate_EmptyK(repeats, testN, len(lbds)*2)
    CS3, COV3, SIZE3 = generate_EmptyK(repeats, testN, 2)
    CS4, COV4, SIZE4 = generate_EmptyK(repeats, testN, 2)
    CS5, COV5, SIZE5 = generate_EmptyK(repeats, testN, 2)
    CS6, COV6, SIZE6 = generate_EmptyK(repeats, testN, 2)
    CS7, COV7, SIZE7 = generate_EmptyK(repeats, testN, 2)
    IND = np.zeros((repeats, testN), dtype=int)
    fullIND = np.zeros((agent_target.n, len(X_test_group)), dtype=int)
    setseed(seed)
    for i in range(len(X_test_group)):
        n_group = X_test_group[i]
        km_ = splitX(agent_target.getX(), n_group)
        fullIND[:, i] = km_.labels_
    pred_kwarg = {'method':predmethod} if predmethod is not None else {'method':'lr'}

    # source agent
    setseed(seed)
    trAgent, agent_a, _, _ = agent_aux.splitAgent(N // 3)
    predAgent, _, _, _ = agent_a.splitAgent(min(N - N // 3, agent_a.n))
    # training predictor on source
    pred = Predictor(**pred_kwarg)
    pred.trainFromAgent(trAgent)
    predAgent.calScore(pred, defaultScore)
    # base generator
    generator_base = Generator(d, hidden_dim, noise_dim)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, epoches, 5e-3, mute=True)

    for rep in range(repeats):
        seed_rep = seed + 1 + rep
        setseed(seed_rep)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#" * 20 + f" Repeat {rep + 1} Started at time {tn} " + f"#" * 20, path=Lpath, islog=isLog)
        setseed(seed_rep)
        # target agent
        testAgent, agent_tar, IND[rep, :], _ = agent_target.splitAgent(testN)
        agent_semi, agent_tar, _, _ = agent_tar.splitAgent(m)
        calTrAgent, calAgent, _, _ = agent_tar.splitAgent(n // 2)
        semiX = agent_semi.getX()
        # oracle data
        calOrac = combineAgents([agent_semi, calAgent, testAgent])

        setseed(seed_rep)
        # training predictor on target
        predictor = Predictor(**pred_kwarg)
        if comb_aux_tr:
            predictor.trainFromAgent(combineAgents([trAgent, calTrAgent]))
        else:
            predictor.trainFromAgent(calTrAgent)
        calTrAgent.calScore(pred, defaultScore)
        calAgent.calScore(predictor, defaultScore)
        calOrac.calScore(predictor, defaultScore)
        setseed(seed_rep)
        generator = deepcopy(generator_base)
        if kwargs.get('cal', None) is not None:
            generator.noise_scalar = kwargs.get('cal')
        else:
            generator.cal_scalar(calTrAgent.getX(), calTrAgent.getS(), 200, stat_type='CvM')
        qrmodel = QRModel('CDF', 1-alpha, CDFModel=generator, d=d)
        setseed(seed_rep)

        # GLCP, CQR
        mar1, size1, mar_std1, size_std1 = [0]*2, [0]*2, [0]*2, [0]*2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                glcp = GLCP(calAgent, deepcopy(generator), predictor, 200, alpha)
                CS1[0, rep], _ = glcp.predict(testAgent.getX(), defaultSolveScore)
            else:
                scc = SCC(calAgent, deepcopy(qrmodel), predictor, alpha)
                CS1[1, rep], _ = scc.predict(testAgent.getX(), defaultSolveScore)
            COV1[i, rep, :] = (CS1[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS1[i, rep, :, 1])
            SIZE1[i, rep, :] = CS1[i, rep, :, 1] - CS1[i, rep, :, 0]
            mar1[i], size1[i], mar_std1[i], size_std1[i] = np.mean(COV1[i, rep, :]), np.mean(SIZE1[i, rep, :]), np.std(np.mean(COV1[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE1[i, :rep + 1, :], axis=-1))

        # SLCP use base score
        l_ = len(lbds)
        n_grid, temperature = int(n_grids[0]), temperatures[0]
        mar2, size2, mar_std2, size_std2 = np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds))
        mar3, size3, mar_std3, size_std3, sel_lbd = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                slcp = SLCP(calAgent, semiX, deepcopy(generator), predictor, [2])
                tuner_list = slcp.tune_lbd_list(5, 100, 5e-3, n_grid, lbds, temperature=temperature, tol_gap=0.001, max_iter=10000, m=200, targ_alpha=(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n), penalty='MSE')
                _, q_list, idx = slcp.auto_lbd_tune(5, 100, 5e-3, alpha, n_grid, lbds, temperature=temperature, gap=tol_gap, tol_gap=0.001, max_iter=10000, m=200, targ_alpha=(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n), penalty='MSE', tuner_list=tuner_list)
            else:
                slcp = SLCP_SCC(calAgent, semiX, deepcopy(generator), deepcopy(qrmodel), predictor, [2])
                tuner_list = slcp.tune_lbd_list(5, 100, 5e-3, n_grid, lbds, tol_gap=0.005, max_iter=10000, targ_alpha=(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n), penalty='MSE')
                _, qhat_list, idx = slcp.auto_lbd_tune(5, 100, 5e-3, alpha, n_grid, lbds, gap=tol_gap, tol_gap=0.005, max_iter=10000, targ_alpha=(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n), penalty='MSE', tuner_list=tuner_list)
            sel_lbd[i] = lbds[idx]
            CS3[i, rep], _ = slcp.predict(testAgent.getX(), defaultSolveScore)
            COV3[i, rep, :] = (CS3[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS3[i, rep, :, 1])
            SIZE3[i, rep, :] = CS3[i, rep, :, 1] - CS3[i, rep, :, 0]
            mar3[i], size3[i], mar_std3[i], size_std3[i] = np.mean(COV3[i, rep, :]), np.mean(SIZE3[i, rep, :]), np.std(np.mean(COV3[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE3[i, :rep + 1, :], axis=-1))
            for param_idx in range(len(lbds)):
                curr_idx = i * l_ + param_idx
                slcp.load_tuner(tuner_list[param_idx], m=200, n=5, temperature=temperature, alpha=alpha)
                CS2[curr_idx, rep], _ = slcp.predict(testAgent.getX(), defaultSolveScore)
                COV2[curr_idx, rep, :] = (CS2[curr_idx, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS2[curr_idx, rep, :, 1])
                SIZE2[curr_idx, rep, :] = CS2[curr_idx, rep, :, 1] - CS2[curr_idx, rep, :, 0]
                mar2[curr_idx], size2[curr_idx], mar_std2[curr_idx], size_std2[curr_idx] = np.mean(COV2[curr_idx, rep, :]), np.mean(SIZE2[curr_idx, rep, :]), np.std(np.mean(COV2[curr_idx, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE2[curr_idx, :rep + 1, :], axis=-1))

        # semi distributional learning
        mar4, size4, mar_std4, size_std4 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                glcp = dissemiGLCP(calAgent, deepcopy(generator), predictor, semiX, 200, alpha, hiddenDim=hidden_dim, batch_size=32, epochs=epoches, learning_rate=5e-3, m=200, iftune=True)
                CS4[0, rep], _ = glcp.predict(testAgent.getX(), defaultSolveScore)
            else:
                scc = dissemiSCC(calAgent, deepcopy(qrmodel), predictor, semiX, alpha, hiddenDim=hidden_dim, batch_size=32, epochs=epoches, learning_rate=5e-3, m=200, iftune=True)
                CS4[1, rep], _ = scc.predict(testAgent.getX(), defaultSolveScore)
            COV4[i, rep, :] = (CS4[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS4[i, rep, :, 1])
            SIZE4[i, rep, :] = CS4[i, rep, :, 1] - CS4[i, rep, :, 0]
            mar4[i], size4[i], mar_std4[i], size_std4[i] = np.mean(COV4[i, rep, :]), np.mean(SIZE4[i, rep, :]), np.std(np.mean(COV4[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE4[i, :rep + 1, :], axis=-1))

        # PPI results
        mar5, size5, mar_std5, size_std5 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                outer_beta = GLCP(predAgent, deepcopy(generator), predictor, 200, alpha).beta
                glcp = dissemiGLCP(calAgent, deepcopy(generator), predictor, semiX, 200, alpha, outer=True, outerX=predAgent.getX(), outerbeta=outer_beta, hiddenDim=hidden_dim, batch_size=32, epochs=20, learning_rate=5e-3, m=200, iftune=True)
                CS5[0, rep], _ = glcp.predict(testAgent.getX(), defaultSolveScore)
            else:
                outer_beta = SCC(predAgent, deepcopy(qrmodel), predictor, alpha).beta
                scc = dissemiSCC(calAgent, deepcopy(qrmodel), predictor, semiX, alpha, outer=True, outerX=predAgent.getX(), outerbeta=outer_beta, hiddenDim=hidden_dim, batch_size=32, epochs=20, learning_rate=5e-3, m=200, iftune=True)
                CS5[1, rep], _ = scc.predict(testAgent.getX(), defaultSolveScore)
            COV5[i, rep, :] = (CS5[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS1[i, rep, :, 1])
            SIZE5[i, rep, :] = CS5[i, rep, :, 1] - CS5[i, rep, :, 0]
            mar5[i], size5[i], mar_std5[i], size_std5[i] = np.mean(COV5[i, rep, :]), np.mean(SIZE5[i, rep, :]), np.std(np.mean(COV5[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE5[i, :rep + 1, :], axis=-1))

        # oracle results
        mar6, size6, mar_std6, size_std6 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                glcp = GLCP(calOrac, deepcopy(generator), predictor, 500, alpha)
                CS6[0, rep], _ = glcp.predict(testAgent.getX(), defaultSolveScore)
            else:
                scc = SCC(calOrac, deepcopy(qrmodel), predictor, alpha)
                CS6[1, rep], _ = scc.predict(testAgent.getX(), defaultSolveScore)
            COV6[i, rep, :] = (CS6[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS1[i, rep, :, 1])
            SIZE6[i, rep, :] = CS6[i, rep, :, 1] - CS6[i, rep, :, 0]
            mar6[i], size6[i], mar_std6[i], size_std6[i] = np.mean(COV6[i, rep, :]), np.mean(SIZE6[i, rep, :]), np.std(np.mean(COV6[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE6[i, :rep + 1, :], axis=-1))

        # SLCP without alignment
        mar7, size7, mar_std7, size_std7 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                load_q = np.quantile(generator.cdf(semiX, generator.generaten(semiX, 5), 200).reshape(-1), (1-alpha)*(calAgent.n + 1) / calAgent.n)
                glcp_noal = GLCP(calAgent, deepcopy(generator), predictor, 200, alpha, loadq=True, q=load_q)
                CS7[0, rep], _ = glcp_noal.predict(testAgent.getX(), defaultSolveScore)
            else:
                load_qhat = np.quantile((generator.generaten(semiX, 5) - qrmodel.predict(semiX)).reshape(-1), (1-alpha)*(calAgent.n + 1) / calAgent.n)
                scc_noal = SCC(calAgent, deepcopy(qrmodel), predictor, alpha, loadq=True, qhat=load_qhat)
                CS7[1, rep], _ = scc_noal.predict(testAgent.getX(), defaultSolveScore)
            COV7[i, rep, :] = (CS7[i, rep, :, 0] <= testAgent.getY().reshape(-1)) & (testAgent.getY().reshape(-1) <= CS7[i, rep, :, 1])
            SIZE7[i, rep, :] = CS7[i, rep, :, 1] - CS7[i, rep, :, 0]
            mar7[i], size7[i], mar_std7[i], size_std7[i] = np.mean(COV7[i, rep, :]), np.mean(SIZE7[i, rep, :]), np.std(np.mean(COV7[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE7[i, :rep + 1, :], axis=-1))

        log(f"REP {rep + 1} (mar, size, mar std, size std)", path=Lpath, islog=isLog)
        for i in range(2):
            name_ = 'GLCP' if i == 0 else 'SCC'
            log(f" --- {name_}: ({mar1[i]:.4f}, {size1[i]:.4f}, {mar_std1[i]:.4f}, {size_std1[i]:.4f})", path=Lpath, islog=isLog)
            for param_idx in range(len(lbds)):
                curr_idx = i * l_ + param_idx
                lbd = lbds[param_idx]
                log(f" ---- SL {i}: ({lbd}) ({mar2[curr_idx]:.4f}, {size2[curr_idx]:.4f}, {mar_std2[curr_idx]:.4f}, {size_std2[curr_idx]:.4f})", path=Lpath, islog=isLog)
            log(f" --- SL-sel {name_}: ({mar3[i]:.4f}, {size3[i]:.4f}, {mar_std3[i]:.4f}, {size_std3[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- SD {name_}: ({mar4[i]:.4f}, {size4[i]:.4f}, {mar_std4[i]:.4f}, {size_std4[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- PPI {name_}: ({mar5[i]:.4f}, {size5[i]:.4f}, {mar_std5[i]:.4f}, {size_std5[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- OR {name_}: ({mar6[i]:.4f}, {size6[i]:.4f}, {mar_std6[i]:.4f}, {size_std6[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- No-al {name_}: ({mar7[i]:.4f}, {size7[i]:.4f}, {mar_std7[i]:.4f}, {size_std7[i]:.4f})", path=Lpath, islog=isLog)

    resDict = {}
    for j in range(2):
        name_ = 'GLCP' if j == 0 else 'SCC'
        mar1, mar_std1, size1, size_std1, local_cov1 = summation_real(COV1[j], SIZE1[j], IND, fullIND, alpha)
        mar2, mar_std2, size2, size_std2, local_cov2 = np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros((len(param_comb), len(X_test_group)))
        for i in range(len(param_comb)):
            mar2[i], mar_std2[i], size2[i], size_std2[i], local_cov2[i] = summation_real(COV2[i+j*len(param_comb)], SIZE2[i+j*len(param_comb)], IND, fullIND, alpha)
        mar3, mar_std3, size3, size_std3, local_cov3 = summation_real(COV3[j], SIZE3[j], IND, fullIND, alpha)
        mar4, mar_std4, size4, size_std4, local_cov4 = summation_real(COV4[j], SIZE4[j], IND, fullIND, alpha)
        mar5, mar_std5, size5, size_std5, local_cov5 = summation_real(COV5[j], SIZE5[j], IND, fullIND, alpha)
        mar6, mar_std6, size6, size_std6, local_cov6 = summation_real(COV6[j], SIZE6[j], IND, fullIND, alpha)
        mar7, mar_std7, size7, size_std7, local_cov7 = summation_real(COV7[j], SIZE7[j], IND, fullIND, alpha)
        resDict = resDict | {f'{name_}': [mar1, mar_std1, size1, size_std1, local_cov1],
                    f'{name_} SDCP': [mar4, mar_std4, size4, size_std4, local_cov4],
                    f'{name_} PPI': [mar5, mar_std5, size5, size_std5, local_cov5],
                    f'{name_} SLCP': [mar2, mar_std2, size2, size_std2, local_cov2],
                    f'{name_} SLCP-sel': [mar3, mar_std3, size3, size_std3, local_cov3],
                    f'{name_} ORCP': [mar6, mar_std6, size6, size_std6, local_cov6],
                    f'{name_} NOAL': [mar7, mar_std7, size7, size_std7, local_cov7],
                    }
        log("-" * 20 + f" Target Results " + "-" * 20, path=Lpath, islog=isLog)
        for p in [Lpath, SumLpath]:
            logResult(mar1, mar_std1, size1, size_std1, local_cov1, f'{name_}', isLog, path=p, log=log)
            logResultParam(mar2, mar_std2, size2, size_std2, local_cov2, f'{name_} SLCP', param_comb, isLog, path=p, log=log)
            logResult(mar3, mar_std3, size3, size_std3, local_cov3, f'{name_} SLCP-sel', isLog, path=p, log=log)
            logResult(mar4, mar_std4, size4, size_std4, local_cov4, f'{name_} SDCP', isLog, path=p, log=log)
            logResult(mar5, mar_std5, size5, size_std5, local_cov5, f'{name_} PPI', isLog, path=p, log=log)
            logResult(mar6, mar_std6, size6, size_std6, local_cov6, f'{name_} ORCP', isLog, path=p, log=log)
            logResult(mar7, mar_std7, size7, size_std7, local_cov7, f'{name_} NOAL', isLog, path=p, log=log)
            log("\n\n", path=Lpath, islog=isLog)
            sum_compare_result(resDict, name_, path=p, isLog=isLog, tol=0.02, n=n//2)

    with open(f'{SimRpath}/{SimName}.pkl', 'wb') as f:
        pickle.dump(resDict, f)

def procedure_cls(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, typeNum, hidden_dim, noise_dim, epoches, alpha, agent_target:Agent, agent_aux:Agent, tol_gap:float=2e-2, in_channels=1, calScoreCl_img=calScoreCl_img, Cl_solveScore=Cl_solveScore, seed=0, model_path='', Lpath='', SumLpath='', SimRpath='', SimName='', isLog=False, addDict=None, **kwargs):
    if isLog:
        assertDir(SimRpath, False)
        assertDir(Lpath, True)
        assertDir(SumLpath, True)
        relog(SumLpath)
        relog(Lpath)
        dic = {'labeled target data': n,
               'unlabeled target data': m,
               'test data': testN,
               'labeled source data': N,
               'seed': seed, 'alpha': alpha, 'repeats': repeats,
               'Engression training epoches': epoches,
               'Engression hidden_dim': hidden_dim,
               "Engression noise_dim": noise_dim,
               "Engression tuning quantile grid number": n_grids,
               "Engression tuning penalty coefficient lambda": lbds,
               "Engression tuning smoothing temperature": temperatures,
               } | addDict
        logInfo(dic, Lpath)
    param_comb = list(itertools.product(n_grids, lbds, temperatures))
    CS1, COV1, SIZE1 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    CS2, COV2, SIZE2 = generate_EmptyK_Cl(repeats, testN, 2*len(param_comb), typeNum)
    CS3, COV3, SIZE3 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    CS4, COV4, SIZE4 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    CS5, COV5, SIZE5 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    CS6, COV6, SIZE6 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    CS7, COV7, SIZE7 = generate_EmptyK_Cl(repeats, testN, 2, typeNum)
    IND = np.zeros((repeats, testN), dtype=int)
    setseed(seed)
    fullIND = agent_target.getY().reshape(-1)
    # source agent
    N = min(N, agent_aux.n//2)
    img_trAgent, img_agent_a, _, _ = agent_aux.splitAgent(N)
    img_predAgent, _, _, _ = img_agent_a.splitAgent(N)
    # training on source
    predictor = MNISTTrainer(batch_size=64, learning_rate=0.001, num_epochs=5, num_classes=typeNum, in_channels=in_channels)
    predictor.load(model_path)
    predictor.prepare_data(train_images=img_trAgent.X, train_labels=img_trAgent.Y)
    predictor.run_training_classifier2(mute=True)
    img_predAgent.loadScore(calScoreCl_img(predictor, img_predAgent.getX(), img_predAgent.getY()))
    predAgent = Agent(d, img_predAgent.n, predictor.covx(img_predAgent.getX()), img_predAgent.getY())
    predAgent.loadScore(img_predAgent.getS())
    # base generator
    generator_base = Generator(d, hidden_dim, noise_dim)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, epoches, 5e-3, mute=True)
    generator_base.noise_scalar = kwargs.get('cal')

    for rep in range(repeats):
        seed_rep = seed + 1 + rep
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#" * 20 + f" Repeat {rep + 1} Started at time {tn} " + f"#" * 20, path=Lpath, islog=isLog)

        setseed(seed_rep)
        # target agent
        img_testAgent, img_agent_tar, IND[rep, :], _ = agent_target.splitAgent(testN)
        img_agent_semi, img_agent_tar, _, _ = img_agent_tar.splitAgent(m)
        img_calAgent, img_agent_tar, _, _ = img_agent_tar.splitAgent(n)
        img_semiX = img_agent_semi.getX()
        img_calOrac = combineImgAgents([img_agent_semi, img_calAgent, img_testAgent])

        setseed(seed_rep)
        img_calAgent.loadScore(calScoreCl_img(predictor, img_calAgent.getX(), img_calAgent.getY()))
        img_calOrac.loadScore(calScoreCl_img(predictor, img_calOrac.getX(), img_calOrac.getY()))

        calAgent = Agent(d, img_calAgent.n, predictor.covx(img_calAgent.getX()), img_calAgent.getY())
        calAgent.loadScore(img_calAgent.getS())
        calOrac = Agent(d, img_calOrac.n, predictor.covx(img_calOrac.getX()), img_calOrac.getY())
        calOrac.loadScore(img_calOrac.getS())
        testAgent = Agent(d, img_testAgent.n, predictor.covx(img_testAgent.getX()), img_testAgent.getY())
        semiX = predictor.covx(img_semiX)
        setseed(seed_rep)
        generator = deepcopy(generator_base)
        qrmodel = QRModel('CDF', 1 - alpha, CDFModel=generator, d=d)

        # GLCP, CQR
        mar1, size1, mar_std1, size_std1 = [0]*2, [0]*2, [0]*2, [0]*2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                glcp = GLCP(calAgent, deepcopy(generator), predictor, 100, alpha)
                CS1[0, rep], _ = glcp.predict(testAgent.getX(), Cl_solveScore)
            else:
                scc = SCC(calAgent, deepcopy(qrmodel), predictor, alpha)
                CS1[1, rep], _ = scc.predict(testAgent.getX(), Cl_solveScore)
            COV1[i, rep, :] = CS1[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE1[i, rep, :] = CS1[i, rep].sum(-1)
            mar1[i], size1[i], mar_std1[i], size_std1[i] = np.mean(COV1[i, rep, :]), np.mean(SIZE1[i, rep, :]), np.std(np.mean(COV1[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE1[i, :rep + 1, :], axis=-1))

        # SLCP use base score
        l_ = len(lbds)
        cov_ = (1 - alpha) * (calAgent.n + 1) / calAgent.n
        n_grid, temperature = int(n_grids[0]), temperatures[0]
        mar2, size2, mar_std2, size_std2 = np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds)), np.zeros(2 * len(lbds))
        mar3, size3, mar_std3, size_std3, sel_lbd = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                slcp = SLCP(calAgent, semiX, deepcopy(generator), predictor, [2])
                tuner_list = slcp.tune_lbd_list(5, 50, 5e-3, n_grid, lbds, temperature=temperature, tol_gap=0.001, max_iter=10000, m=100, targ_alpha=1 - cov_, penalty='MSE')
                _, q_list, idx = slcp.auto_lbd_tune(5, 50, 5e-3, alpha, n_grid, lbds, temperature=temperature, gap=tol_gap, tol_gap=0.001, max_iter=10000, m=100, targ_alpha=1 - cov_, penalty='MSE', tuner_list=tuner_list)
            else:
                slcp = SLCP_SCC(calAgent, semiX, deepcopy(generator), deepcopy(qrmodel), predictor, [2])
                tuner_list = slcp.tune_lbd_list(5, 50, 5e-3, n_grid, lbds, tol_gap=0.001, max_iter=10000, targ_alpha=1 - cov_, penalty='MSE')
                _, qhat_list, idx = slcp.auto_lbd_tune(5, 50, 5e-3, alpha, n_grid, lbds, gap=tol_gap, tol_gap=0.001, max_iter=10000, targ_alpha=1 - cov_, penalty='MSE', tuner_list=tuner_list)
            sel_lbd[i] = lbds[idx]
            CS3[i, rep], _ = slcp.predict(testAgent.getX(), Cl_solveScore)
            COV3[i, rep, :] = CS3[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE3[i, rep, :] = CS3[i, rep].sum(-1)
            mar3[i], size3[i], mar_std3[i], size_std3[i] = np.mean(COV3[i, rep, :]), np.mean(SIZE3[i, rep, :]), np.std(np.mean(COV3[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE3[i, :rep + 1, :], axis=-1))
            for param_idx in range(len(lbds)):
                curr_idx = i*l_+param_idx
                slcp.load_tuner(tuner_list[param_idx], m=100, n=5, temperature=temperature, alpha=alpha)
                CS2[curr_idx, rep], _ = slcp.predict(testAgent.getX(), Cl_solveScore)
                COV2[curr_idx, rep, :] = CS2[curr_idx, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
                SIZE2[curr_idx, rep, :] = CS2[curr_idx, rep].sum(-1)
                mar2[curr_idx], size2[curr_idx], mar_std2[curr_idx], size_std2[curr_idx] = np.mean(COV2[curr_idx, rep, :]), np.mean(SIZE2[curr_idx, rep, :]), np.std(np.mean(COV2[curr_idx, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE2[curr_idx, :rep + 1, :], axis=-1))

        # semi distributional learning
        mar4, size4, mar_std4, size_std4 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed + 1 + rep)
            if i == 0:
                glcp = dissemiGLCP(calAgent, deepcopy(generator), predictor, semiX, 100, alpha, hiddenDim=hidden_dim, batch_size=32, epochs=epoches, learning_rate=5e-3, m=100, iftune=True)
                CS4[0, rep], _ = glcp.predict(testAgent.getX(), Cl_solveScore)
            else:
                scc = dissemiSCC(calAgent, deepcopy(qrmodel), predictor, semiX, alpha, hiddenDim=hidden_dim, batch_size=32, epochs=epoches, learning_rate=5e-3, m=100, iftune=True)
                CS4[1, rep], _ = scc.predict(testAgent.getX(), Cl_solveScore)
            COV4[i, rep, :] = CS4[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE4[i, rep, :] = CS4[i, rep].sum(-1)
            mar4[i], size4[i], mar_std4[i], size_std4[i] = np.mean(COV4[i, rep, :]), np.mean(SIZE4[i, rep, :]), np.std(np.mean(COV4[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE4[i, :rep + 1, :], axis=-1))

        # PPI results
        mar5, size5, mar_std5, size_std5 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed + 1 + rep)
            if i == 0:
                outer_beta = GLCP(predAgent, deepcopy(generator), predictor, 100, alpha).beta
                glcp = dissemiGLCP(calAgent, deepcopy(generator), predictor, semiX, 100, alpha, outer=True, outerX=predAgent.getX(), outerbeta=outer_beta, hiddenDim=hidden_dim, batch_size=32, epochs=20, learning_rate=5e-3, m=100, iftune=True)
                CS5[0, rep], _ = glcp.predict(testAgent.getX(), Cl_solveScore)
            else:
                outer_beta = SCC(predAgent, deepcopy(qrmodel), predictor, alpha).beta
                scc = dissemiSCC(calAgent, deepcopy(qrmodel), predictor, semiX, alpha, outer=True, outerX=predAgent.getX(), outerbeta=outer_beta, hiddenDim=hidden_dim, batch_size=32, epochs=20, learning_rate=5e-3, m=100, iftune=True)
                CS5[1, rep], _ = scc.predict(testAgent.getX(), Cl_solveScore)
            COV5[i, rep, :] = CS5[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE5[i, rep, :] = CS5[i, rep].sum(-1)
            mar5[i], size5[i], mar_std5[i], size_std5[i] = np.mean(COV5[i, rep, :]), np.mean(SIZE5[i, rep, :]), np.std(np.mean(COV5[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE5[i, :rep + 1, :], axis=-1))

        # Oracle results
        mar6, size6, mar_std6, size_std6 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed + 1 + rep)
            if i == 0:
                glcp = GLCP(calOrac, deepcopy(generator), predictor, 200, alpha)
                CS6[0, rep], _ = glcp.predict(testAgent.getX(), Cl_solveScore)
            else:
                scc = SCC(calOrac, deepcopy(qrmodel), predictor, alpha)
                CS6[1, rep], _ = scc.predict(testAgent.getX(), Cl_solveScore)
            COV6[i, rep, :] = CS6[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE6[i, rep, :] = CS6[i, rep].sum(-1)
            mar6[i], size6[i], mar_std6[i], size_std6[i] = np.mean(COV6[i, rep, :]), np.mean(SIZE6[i, rep, :]), np.std(np.mean(COV6[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE6[i, :rep + 1, :], axis=-1))

        # SLCP without alignment
        mar7, size7, mar_std7, size_std7 = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        for i in range(2):
            setseed(seed_rep)
            if i == 0:
                load_q = np.quantile(generator.cdf(semiX, generator.generaten(semiX, 5), 200).reshape(-1), (1-alpha)*(calAgent.n + 1) / calAgent.n)
                glcp_noal = GLCP(calAgent, deepcopy(generator), predictor, 200, alpha, loadq=True, q=load_q)
                CS7[0, rep], _ = glcp_noal.predict(testAgent.getX(), Cl_solveScore)
            else:
                load_qhat = np.quantile((generator.generaten(semiX, 5) - qrmodel.predict(semiX)).reshape(-1), (1-alpha)*(calAgent.n + 1) / calAgent.n)
                scc_noal = SCC(calAgent, deepcopy(qrmodel), predictor, alpha, loadq=True, qhat=load_qhat)
                CS7[1, rep], _ = scc_noal.predict(testAgent.getX(), Cl_solveScore)
            COV7[i, rep, :] = CS7[i, rep, np.arange(testAgent.n), np.astype(testAgent.getY().reshape(-1), np.int16)]
            SIZE7[i, rep, :] = CS7[i, rep].sum(-1)
            mar7[i], size7[i], mar_std7[i], size_std7[i] = np.mean(COV7[i, rep, :]), np.mean(SIZE7[i, rep, :]), np.std(np.mean(COV7[i, :rep + 1, :], axis=-1)), np.std(np.mean(SIZE7[i, :rep + 1, :], axis=-1))

        log(f"REP {rep + 1} (mar, size, mar std, size std)", path=Lpath, islog=isLog)
        for i in range(2):
            name_ = 'GLCP' if i == 0 else 'SCC'
            log(f" --- {name_}: ({mar1[i]:.4f}, {size1[i]:.4f}, {mar_std1[i]:.4f}, {size_std1[i]:.4f})", path=Lpath, islog=isLog)
            for param_idx in range(len(param_comb)):
                curr_idx = i * l_ + param_idx
                n_grid, lbd, temperature = param_comb[param_idx]
                log(f" ---- SL: ({lbd}) ({mar2[curr_idx]:.4f}, {size2[curr_idx]:.4f}, {mar_std2[curr_idx]:.4f}, {size_std2[curr_idx]:.4f})", path=Lpath, islog=isLog)
            log(f" --- SL-sel {name_}: ({mar3[i]:.4f}, {size3[i]:.4f}, {mar_std3[i]:.4f}, {size_std3[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- SD {name_}: ({mar4[i]:.4f}, {size4[i]:.4f}, {mar_std4[i]:.4f}, {size_std4[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- PPI {name_}: ({mar5[i]:.4f}, {size5[i]:.4f}, {mar_std5[i]:.4f}, {size_std5[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- OR {name_}: ({mar6[i]:.4f}, {size6[i]:.4f}, {mar_std6[i]:.4f}, {size_std6[i]:.4f})", path=Lpath, islog=isLog)
            log(f" --- No-al {name_}: ({mar7[i]:.4f}, {size7[i]:.4f}, {mar_std7[i]:.4f}, {size_std7[i]:.4f})", path=Lpath, islog=isLog)

    resDict = {}
    for j in range(2):
        name_ = 'GLCP' if j == 0 else 'SCC'
        mar1, mar_std1, size1, size_std1, local_cov1 = summation_real_cls(COV1[j], SIZE1[j], IND, fullIND, alpha)
        mar2, mar_std2, size2, size_std2, local_cov2 = np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros(len(param_comb)), np.zeros((len(param_comb)))
        for i in range(len(param_comb)):
            mar2[i], mar_std2[i], size2[i], size_std2[i], local_cov2[i] = summation_real_cls(COV2[i+j*len(param_comb)], SIZE2[i+j*len(param_comb)], IND, fullIND, alpha)
        mar3, mar_std3, size3, size_std3, local_cov3 = summation_real_cls(COV3[j], SIZE3[j], IND, fullIND, alpha)
        mar4, mar_std4, size4, size_std4, local_cov4 = summation_real_cls(COV4[j], SIZE4[j], IND, fullIND, alpha)
        mar5, mar_std5, size5, size_std5, local_cov5 = summation_real_cls(COV5[j], SIZE5[j], IND, fullIND, alpha)
        mar6, mar_std6, size6, size_std6, local_cov6 = summation_real_cls(COV6[j], SIZE6[j], IND, fullIND, alpha)
        mar7, mar_std7, size7, size_std7, local_cov7 = summation_real_cls(COV7[j], SIZE7[j], IND, fullIND, alpha)
        resDict = resDict | {f'{name_}': [mar1, mar_std1, size1, size_std1, local_cov1],
                    f'{name_} SDCP': [mar4, mar_std4, size4, size_std4, local_cov4],
                    f'{name_} PPI': [mar5, mar_std5, size5, size_std5, local_cov5],
                    f'{name_} SLCP': [mar2, mar_std2, size2, size_std2, local_cov2],
                    f'{name_} SLCP-sel': [mar3, mar_std3, size3, size_std3, local_cov3],
                    f'{name_} ORCP': [mar6, mar_std6, size6, size_std6, local_cov6],
                    f'{name_} NOAL': [mar7, mar_std7, size7, size_std7, local_cov7],
                    }
        log("-" * 20 + f" Target Results " + "-" * 20, path=Lpath, islog=isLog)
        for p in [Lpath, SumLpath]:
            logResult_cls(mar1, mar_std1, size1, size_std1, local_cov1, f'{name_}', isLog, path=p, log=log)
            logResultParam_cls(mar2, mar_std2, size2, size_std2, local_cov2, f'{name_} SLCP', param_comb, isLog, path=p, log=log)
            logResult_cls(mar3, mar_std3, size3, size_std3, local_cov3, f'{name_} SLCP-sel', isLog, path=p, log=log)
            logResult_cls(mar4, mar_std4, size4, size_std4, local_cov4, f'{name_} SDCP', isLog, path=p, log=log)
            logResult_cls(mar5, mar_std5, size5, size_std5, local_cov5, f'{name_} PPI', isLog, path=p, log=log)
            logResult_cls(mar6, mar_std6, size6, size_std6, local_cov6, f'{name_} ORCP', isLog, path=p, log=log)
            logResult_cls(mar7, mar_std7, size7, size_std7, local_cov7, f'{name_} NOAL', isLog, path=p, log=log)
            log("\n\n", path=Lpath, islog=isLog)
            sum_compare_result(resDict, name_, path=p, isLog=isLog, tol=0.02, n=n//2, cls=True)

    with open(f'{SimRpath}/{SimName}.pkl', 'wb') as f:
        pickle.dump(resDict, f)
