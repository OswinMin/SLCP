from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
import pickle

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def log(s, path, islog):
    if islog:
        with open(path, 'a') as f:
            f.write(s+'\n')
    else:
        print(s)

def generate_EmptyK(repeats, testN, k):
    CS, COV, SIZE = np.zeros((k, repeats, testN, 2)), np.zeros((k, repeats, testN)), np.zeros((k, repeats, testN))
    return CS, COV, SIZE

def generate_EmptyK_Cl(repeats, testN, k, typeNum):
    CS, COV, SIZE = np.zeros((k, repeats, testN, typeNum)), np.zeros((k, repeats, testN)), np.zeros((k, repeats, testN))
    return CS, COV, SIZE

def generate_Empty(repeats, testN):
    CS, COV, SIZE = np.zeros((repeats, testN, 2)), np.zeros((repeats, testN)), np.zeros((repeats, testN))
    return CS, COV, SIZE

def generate_Empty_Cl(repeats, testN, typeNum):
    CS, COV, SIZE = np.zeros((repeats, testN, typeNum)), np.zeros((repeats, testN)), np.zeros((repeats, testN))
    return CS, COV, SIZE

def setseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def checkDir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass

def relog(path):
    with open(path, 'w') as f:
        f.write('')

def logInfo(dic, Lpath):
    for k, v in dic.items():
        log(f"{k} : {v}", path=Lpath, islog=True)

def identityMap(fx, y):
    return y.reshape((-1,1))

def solveIdentity(x, s, predictor):
    return s

def defaultScore(fx, y):
    return np.abs(fx.reshape((-1,1))-y.reshape((-1,1)))

def defaultSolveScore(x, s, predictor):
    """
    :param x: [d]
    :param s:
    :param predictor:
    :return:
    """
    if len(x.shape) == 1:
        yhat = predictor.predict(x).item()
        return np.array([yhat - s, yhat + s]), False
    else:
        yhat = predictor.predict(x).reshape(-1)
        cs = np.zeros((yhat.shape[0], 2))
        cs[:, 0] = yhat - s.reshape(-1)
        cs[:, 1] = yhat + s.reshape(-1)
        return cs, np.zeros(x.shape[0], dtype=bool)

def empiricalQuantile(X:np.ndarray, W:np.ndarray, q:float=.9):
    indice = np.argsort(X, axis=1)
    rowindice = np.repeat(np.array(range(indice.shape[0])), W.shape[1], axis=0)
    rowsorted_X = X[rowindice, indice.reshape(-1)].reshape((X.shape[0], X.shape[1]))
    rowsorted_W = W[rowindice, indice.reshape(-1)].reshape((W.shape[0], W.shape[1]))
    cumsum_W = np.cumsum(rowsorted_W, axis=1)
    mask = cumsum_W > q
    re_X = rowsorted_X * mask + (1-mask) * (np.max(X)+1)
    quantile = np.min(re_X, axis=1)
    return quantile

def summation(COV, SIZE, alpha:float=.1):
    """
    :param COV: repeats, testN
    :param SIZE: repeats, testN
    """
    mar = np.mean(COV)
    size = np.mean(SIZE)
    size_std = np.std(np.mean(SIZE, axis=1))
    mar_std = np.std(np.mean(COV, axis=1))
    local_mar = COV.mean(0)
    local_cov = np.mean(np.abs(local_mar-(1-alpha)))
    tt_local_cov = np.mean(np.abs(COV-(1-alpha)))
    return mar, mar_std, size, size_std, local_cov, tt_local_cov

def saveData(path, CS, COV, SIZE, name):
    np.save(f"{path}/{name}_CS.npy", CS)
    np.save(f"{path}/{name}_COV.npy", COV)
    np.save(f"{path}/{name}_SIZE.npy", SIZE)

def saveData_Pickle(path, CS, COV, SIZE, name):
    with open(f"{path}/{name}_CS.pkl", 'wb') as f:
        pickle.dump(CS, f)
    np.save(f"{path}/{name}_COV.npy", COV)
    np.save(f"{path}/{name}_SIZE.npy", SIZE)

def eff_Mat_Dis(A, B):
    """
    :param A: n*d
    :param B: m*d
    :return: n*m L2 distance for i-th row of A and j-th row of B
    """
    normA = (A**2).sum(-1)  # n
    normB = (B**2).sum(-1)  # m
    AdotB = np.dot(A, B.T)  # n*m
    return normA[:, np.newaxis] + normB[np.newaxis, :] - 2 * AdotB

class CDF:
    def __init__(self):
        pass
    def conditional_cdf(self, x:np.ndarray, y:np.ndarray):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return np.ones((x.shape[0], y.shape[1]))

class ModelWrapper:
    def __init__(self, model, device="cpu", **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.device = torch.device(device)
    def predict_img(self, images):
        return
    def predict_feat(self, features):
        return
    def load(self, filepath):
        return
    def save(self, filepath):
        return
    def covx(self, images):
        return