import numpy as np
from tools import *
from Agents import *
from Predictor import *

class SCP:
    def __init__(self, agent:Agent, predictor:Predictor, scorefun:defaultScore, solveScore:defaultSolveScore):
        self.agent = agent
        self.predictor = predictor
        self.scorefun = scorefun
        self.solveScore = solveScore
        self.n = self.agent.n

    def predict(self, testX, alpha, subIndex=None, csLen:int=2):
        if subIndex is None:
            subIndex = np.arange(testX.shape[1])
        else:
            subIndex = np.array(subIndex)
        q = (1-alpha) * (self.n + 1) / self.n
        qhat = np.quantile(self.agent.Score, q, method='higher')
        if len(testX.shape) == 1:
            testX = testX.reshape((1, -1))
        cs = np.zeros((testX.shape[0], csLen))
        for i in range(testX.shape[0]):
            cs[i, :] = self.solveScore(testX[i, subIndex], qhat, self.predictor)
        return cs