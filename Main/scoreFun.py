from __future__ import annotations
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import QuantileRegressor
from engression import engression
from LCP import *
from RLCP import *
from tools import *

class Eng():
    def __init__(self, input_dim: int, num_layer=2, hidden_dim=100, noise_dim=100, w=0.):
        super().__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.w = w

    def train(self, agent:Agent, epoches:int=100, bs:int=32,  lr:float=1e-3, mute:bool=True):
        X = torch.tensor(agent.X).float()
        Y = torch.tensor(agent.Y).float()
        self.lim = [np.min(Y), np.max(Y)]
        if mute:
            with suppress_stdout():
                self.engressor = engression(X, Y, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.noise_dim, lr=lr, num_epochs=epoches, batch_size=bs, device='cpu')
        else:
            self.engressor = engression(X, Y, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.noise_dim, lr=lr, num_epochs=epoches, batch_size=bs, device='cpu')

    def generateN(self, x:np.ndarray, n:int=100) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        x = torch.tensor(x).float()
        y = self.engressor.sample(x, sample_size=n).view(x.shape[0], n).numpy()
        return y

    def percentile(self, x:np.ndarray, s:np.ndarray, n:int=100, maxS:float=-.0, w:np.ndarray=None):
        if w is None:
            w = np.zeros((x.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        preds = self.generateN(x, n)
        weight = np.ones_like(preds)
        weight = weight / weight.shape[1] * (1 - w)
        preds = np.concatenate((preds, np.ones((preds.shape[0],1))*maxS), axis=1)
        weight = np.concatenate((weight, w), axis=1)
        perc = (weight * (preds <= s.reshape((-1,1)))).sum(-1)
        return perc

    def getQuantile(self, X:np.ndarray, q:Union[float,floating]=.9, n:int=100, maxS:float=-.0, w:np.ndarray=None):
        if w is None:
            w = np.zeros((X.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        Yhats = self.generateN(X, n)
        weight = np.ones_like(Yhats)
        Yhats = np.concatenate((Yhats, np.ones((Yhats.shape[0], 1)) * maxS), axis=1)
        weight = weight / weight.shape[1] * (1 - w)
        weight = np.concatenate((weight, w), axis=1)
        qhat = empiricalQuantile(Yhats, weight, q)
        return qhat

    def calW(self, x:np.ndarray, X_C:np.ndarray, ave:float=.02):
        if len(X_C.shape) == 1:
            X_C = X_C.reshape((1,-1))
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        w = ((x-X_C)**2).sum(-1)
        std = w.std()
        w = np.exp(-w/std)
        w = np.clip(w / w.mean() * ave, 0, .1)
        return w

class condPredictor_QR:
    def __init__(self, d:int, nQuan:int=101):
        self.d = d
        self.nQuan = nQuan
        self.tau = np.array(range(1, nQuan)) / nQuan
        self.tau[nQuan//2:] = self.tau[nQuan//2:]+0.95/nQuan
        self.tau[:nQuan//2] = self.tau[:nQuan//2]-0.95/nQuan
        self.beta = np.zeros((self.nQuan-1, self.d+1))

    def train(self, X, Y, alpha=0.):
        for i in range(self.nQuan-1):
            model = QuantileRegressor(quantile=self.tau[i], alpha=alpha, solver='highs')
            model.fit(X, Y.reshape(-1))
            self.beta[i, :-1] = model.coef_
            self.beta[i, -1] = model.intercept_

    def predict(self, X:np.ndarray):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        return X @ self.beta[:, :-1].T + self.beta[:, [-1]].T

def QuanScore(fx, y):
    """
    fx : n * nQuan-1
    y : n * 1
    return : n * 1
    """
    gap = 0.049 / (fx.shape[1]+1)
    return np.abs((fx <= y).mean(-1).reshape((-1,1)) - 0.5) + np.random.uniform(0, gap, (fx.shape[0], 1))

def SolveQuanScore(x, s, predictor:Union(condPredictor_QR)):
    """
    x : d
    predictor : n * d -> n * nQuan-1
    s : float
    """
    fx = np.sort(predictor.predict(x).reshape(-1))   # nQuan-1
    gap = 0.049 / (fx.shape[0]+1)
    n = len(fx)
    if 0.5 <= s:
        u = np.random.uniform(0, gap, 1)
        if u<=(s-0.5)/gap:
            return [np.min(fx), np.max(fx)], True
        else:
            return [np.min(fx), np.max(fx)], False
    i1, i2 = np.ceil(n*(0.5-s)).astype(int)-1, np.floor(n*(0.5+s)).astype(int)
    u1, u2 = np.random.uniform(0, 1, 2)
    x1 = fx[i1] if u1<=(s-np.abs((i1+1)/n-0.5))/gap else fx[i1+1]
    x2 = fx[i2] if u2<=(s-np.abs((i2)/n-0.5))/gap else fx[i2-1]
    return [x1, x2], False
    # return [fx[i1], fx[i2]], False

if __name__ == '__main__':
    def genAgent(k, n, gamma, me, l=-2., u=2.):
        def sigma(X, gamma):
            # return np.sqrt(np.abs(np.cos(X))[:,:5].sum(-1) * gamma)
            return np.sqrt((X[:,:5].sum(-1)/2)**2 * gamma)
            # return np.sqrt(np.cosh(X)[:,:5].sum(-1) * gamma)
        X = np.random.uniform(l, u, (n, k))
        Y = np.random.normal(0, 1, n) * sigma(X, gamma)
        Y = Y.reshape((-1, 1)) + X.sum(-1).reshape((-1, 1)) * me
        agent = Agent(k, n, X, Y)

        def calCov(X, CS, isinf):
            CS = (CS - X.sum(-1).reshape((-1, 1)) * me) / sigma(X, gamma).reshape((-1, 1))
            localCov = stats.norm.cdf(CS[:, 1]) - stats.norm.cdf(CS[:, 0])
            localCov[isinf] = 1.
            return localCov
        agent.loadCalCov(calCov)
        return agent

    d = 10
    kernel = Loo_eff_Mat_Kernel
    nTest = 2000
    loc_cov_elcp, loc_cov_lcp = np.zeros(nTest), np.zeros(nTest)
    setseed(1)
    agentTest = genAgent(d, nTest, 1, 1)
    testX = agentTest.X
    testY = agentTest.Y

    agent = [genAgent(d, 200, 1, 1), genAgent(d, 2000, 4, 2)]
    agentTr = [genAgent(d, 200, 1, 1), genAgent(d, 500, 4, 2)]

    predictors = []
    DREs = [DRE(method='unit')]
    for i in range(len(agent)):
        predictor = condPredictor_QR(d, 101)
        predictor.train(agentTr[i].X, agentTr[i].Y, alpha=1.)
        agent[i].calScore(predictor, QuanScore)
        predictors.append(predictor)
        if i > 0:
            dre = DRE(X=agent[0].getXS(), Z=agent[i].getXS(), method='qda-c', cv=2)
            DREs.append(dre)

    s = np.quantile(agent[0].Score, 0.9)
    covs = []
    for i in range(testX.shape[0]):
        x = testX[i, :]
        cs, isinf = SolveQuanScore(x, s, predictors[0])
        locc = agent[0].calCov(x.reshape((1,-1)), np.array(cs).reshape((1,-1)), isinf)
        covs.append(locc)
    np.abs(np.array(covs)-0.9).mean(), np.mean(covs)

    # elcp = ELCP(agent, predictors, DREs)
    # for h in [0.8,1,1.2,1.4]:
    #     cs, isinf = elcp.ELCP(testX, 0.9, h=h, w=1., kernel=kernel, solveS=SolveQuanScore)
    #     loc_cov_elcp = agent[0].calCov(testX, cs, isinf)
    #     print(f"ELCP {h}: {np.mean(np.abs(loc_cov_elcp - 0.9)):.4f}, {np.mean(loc_cov_elcp):.4f}, {np.mean(isinf):.4f}")
    #
    # lcp = ELCP([agent[0]], [predictors[0]], [DREs[0]])
    # for h in [0.8,1,1.2,1.4]:
    #     cs, isinf = lcp.ELCP(testX, 0.9, h=h, w=0., kernel=kernel, solveS=SolveQuanScore)
    #     loc_cov_lcp = agent[0].calCov(testX, cs, isinf)
    #     print(f"LCP {h}: {np.mean(np.abs(loc_cov_lcp - 0.9)):.4f}, {np.mean(loc_cov_lcp):.4f}, {np.mean(isinf):.4f}")
    #
    # rlcp = RLCP(predictors[0], [agent[0]], True)
    # for h in [0.8, 1, 1.2, 1.4, 2., 3., 4.]:
    #     cs, isinf = rlcp.RLCP(testX, 0.9, h=h, solveS=SolveQuanScore)
    #     loc_cov_rlcp = agent[0].calCov(testX, cs, isinf)
    #     print(f"RLCP {h}: {np.mean(np.abs(loc_cov_rlcp - 0.9)):.4f}, {np.mean(loc_cov_rlcp):.4f}, {np.mean(isinf):.4f}")

    # X = np.random.normal(0, 1, (100, 5))
    # testX = np.random.normal(0, 1, (100, 5))
    # Y = X.sum(-1) + np.random.normal(0, 1)
    # predictor = condPredictor_QR(5)
    # predictor.train(X, Y, alpha=0.05)
    # x = testX[0, :]
    # fx = predictor.predict(x)
    # SolveQuanScore(x, 0.45, predictor)
    # QuanScore(fx, 5.20741260)