from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Union
from tools import *

class Agent():
    """
    创建 Agent 流程：
        初始化加载 X, Y
        .loadScore() 加载计算好的 score
    """
    def __init__(self, d:int, n:int, X:np.ndarray, Y:np.ndarray):
        """
        :param d: input dimension
        :param n: num of observations
        :param X: shape n*d
        :param Y: shape n*1
        """
        super(Agent, self).__init__()
        self.d = d
        self.n = n
        self.X = X
        self.Y = Y.reshape((n, 1))
        self.hasScore = False

    def loadCDF(self, cdf:CDF):
        """
        :param cdf: 要求 cdf 有 .conditional_cdf(x:np.ndarray, y:np.ndarray) 方法，计算 y | x 的条件分布，x : n * d   y : n * k ( k 一般是 1 )
        :return:
        """
        self.CDF = cdf

    def coverage(self, X:np.ndarray, CS:np.ndarray):
        if self.CDF is None:
            print(f"No oracle CDF, fails")
            return
        upper = self.CDF.conditional_cdf(X, CS[:, [1]])
        lower = self.CDF.conditional_cdf(X, CS[:, [0]])
        coverage = upper - lower
        return coverage

    def getData(self):
        return self.X, self.Y

    def loadScore(self, S:np.ndarray):
        assert S.shape[0] == self.n
        self.Score: np.ndarray = S.reshape((-1,1))
        self.hasScore = True

    def calScore(self, predictor:Predictor, scoreFun=defaultScore):
        """
        :param predictor: must implement method 'predict'
                params: x:np.ndarray, n*d
                return: n*1 np.ndarray
        self.Score: np.ndarray n*1
        """
        self.Score:np.ndarray = scoreFun(predictor.predict(self.X), self.Y)
        self.hasScore = True

    def getXS(self, subIndex=None):
        if subIndex is None:
            subIndex = np.arange(self.X.shape[1])
        XS = np.concatenate((self.X[:, subIndex], self.Score), axis=1)
        return XS

    def getXY(self, subIndex=None):
        if subIndex is None:
            subIndex = np.arange(self.X.shape[1])
        XY = np.concatenate((self.X[:, subIndex], self.Y), axis=1)
        return XY

    def getX_T(self):
        return torch.tensor(self.X).float()

    def getY_T(self):
        return torch.tensor(self.Y).float()

    def getS_T(self):
        return torch.tensor(self.Score).float()

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def getS(self):
        return self.Score

    def splitAgent(self, k):
        ind = np.random.permutation(self.n)
        X, Y = self.X[ind], self.Y[ind]
        agent1 = Agent(self.d, k, X[:k], Y[:k])
        agent2 = Agent(self.d, self.n-k, X[k:], Y[k:])
        if self.hasScore:
            S = self.Score[ind]
            agent1.loadScore(S[:k])
            agent2.loadScore(S[k:])
        return agent1, agent2, ind[:k], ind[k:]

def combineAgents(agentList:list[Agent]):
    d = agentList[0].d
    n = 0
    X = np.zeros((0,d))
    Y = np.zeros((0,1))
    S = np.zeros((0,1))
    hasScore = True
    for agent in agentList:
        X = np.vstack([X, agent.X])
        Y = np.vstack([Y, agent.Y])
        try:
            S = np.vstack([S, agent.Score])
        except:
            hasScore = False
        n += agent.n
    agent = Agent(d, n, X, Y)
    if hasScore:
        agent.Score = S
    return agent

