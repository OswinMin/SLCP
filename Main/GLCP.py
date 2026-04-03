import numpy as np
from tools import *
from Agents import *
from engGenerator import *
from Predictor import *
from QuanRegressor import *

class GLCP:
    def __init__(self, targetAgent:Agent, generator:Generator, predictor:Predictor, n: int = 1000, alpha: float = 0.1, **kwargs):
        """
        :param targetAgent:
        :param generator:
        :param predictor:
        """
        self.targetAgent = targetAgent
        self.generator = generator
        self.predictor = predictor
        if kwargs.get('loadq', False):
            self.q = kwargs.get('q', None)
            self.n = n
        else:
            self.quantileOnCalibration(n=n, alpha=alpha)

    def phi(self, beta):
        return np.abs(beta-0.5)

    def quantileOnCalibration(self, n: int = 1000, alpha: float = 0.1):
        self.n = n
        # maxS = np.max(self.targetAgent.getS())
        # self.generator.loadMax(0.005, maxS)
        self.beta = self.generator.cdf(self.targetAgent.getX(), self.targetAgent.getS(), n).reshape(-1)
        betas = np.concatenate([self.beta, [1.]])
        self.q = np.quantile(betas, 1 - alpha, method='higher')

    def predict(self, testX: np.ndarray, solveScore=defaultSolveScore):
        if len(testX.shape) == 1:
            testX = testX.reshape((1, -1))

        sq = self.generator.quantile(testX, self.q, self.n)
        cs, isinf = solveScore(testX, sq, self.predictor)
        if self.q >= 1.:
            isinf = np.ones(testX.shape[0], dtype=bool)
        return cs, isinf

class SCC:
    def __init__(self, targetAgent:Agent, predictor:QRModel, basePredictor:Predictor, alpha: float = 0.1, **kwargs):
        self.targetAgent = targetAgent
        self.predictor = predictor
        self.basePredictor = basePredictor
        if kwargs.get('loadq', False):
            self.qhat = kwargs.get('qhat', None)
        else:
            self.quantileOnCalibration(alpha=alpha)

    def quantileOnCalibration(self, alpha: float = 0.1):
        cov = (1 - alpha) * (self.targetAgent.n + 1) / self.targetAgent.n
        self.beta = self.targetAgent.getS() - self.predictor.predict(self.targetAgent.getX())
        self.qhat = np.quantile(self.beta, cov, method='higher')

    def predict(self, testX, solveScore=defaultSolveScore, **kwargs):
        if len(testX.shape) == 1:
            testX = testX.reshape((1, -1))

        s_cs = np.maximum(self.predictor.predict(testX, **kwargs).reshape(-1) + self.qhat, 0)
        cs, isinf = solveScore(testX, s_cs, self.basePredictor)
        return cs, isinf


if __name__ == '__main__':
    import scipy.stats as ss
    def generateXY(gamma, n, d):
        X = np.random.normal(0, 1, (n, d))
        Y = np.random.normal(0, 1, (n, 1)) * ((np.abs(X).sum(-1)) ** 0.5).reshape((-1, 1)) * gamma
        return d, n, X, Y
    def coverage(interval, x, gamma):
        interval = [num / ((np.abs(x).sum() ** 0.5) * gamma) for num in interval]
        return ss.norm.cdf(interval[1]) - ss.norm.cdf(interval[0])

    setseed(0)
    n = 100
    d = 3
    gamma = np.random.uniform(0.8, 1, 10)
    predTrAgents = [Agent(*generateXY(ga, n//2, d)) for ga in gamma]
    calTrAgents = [Agent(*generateXY(ga, n//2, d)) for ga in gamma]
    predictorList = [Predictor('lr') for _ in predTrAgents]
    for i in range(len(predictorList)):
        predictorList[i].trainFromAgent(predTrAgents[i])
        calTrAgents[i].calScore(predictorList[i], defaultScore)
    calTrAgent = combineAgents(calTrAgents)
    generator = Generator(d, [20,50,20], False, d)
    generator.trainEng(calTrAgent.getX(), calTrAgent.getS(), 10, 32, 300, 0.005, mute=True)

    alpha, g, reps = 0.1, 0.9, 10
    lb = np.zeros((reps, 2))
    for i in range(reps):
        setseed(i+1)
        trg = Agent(*generateXY(g, 30, d))
        calg = Agent(*generateXY(g, 30, d))
        testg = Agent(*generateXY(g, 100, d))
        predictor = Predictor('lr')
        predictor.trainFromAgent(trg)
        calg.calScore(predictor, defaultScore), testg.calScore(predictor, defaultScore)
        testX = testg.getX()

        glcp = GLCP(calg, generator, predictor, 500, alpha)
        cs, isinf = glcp.predict(testX, defaultSolveScore)
        size = cs[:, 1] - cs[:, 0]
        cov = [np.maximum(coverage(cs[i, :], testX[i, :], g), isinf[i]) for i in range(cs.shape[0])]
        print(f"DCP: {np.mean(size):.4f}, {np.mean(cov):.4f}, {glcp.q:.4f}")
        lb[i, :] = np.mean(size), np.mean(cov)
    print(np.mean(lb, axis=0))

    qr = QRModel(methods='CDF', CDFModel=generator, d=d, m=500)
    for i in range(reps):
        setseed(i+1)
        trg = Agent(*generateXY(g, 30, d))
        calg = Agent(*generateXY(g, 30, d))
        testg = Agent(*generateXY(g, 100, d))
        predictor = Predictor('lr')
        predictor.trainFromAgent(trg)
        calg.calScore(predictor, defaultScore), testg.calScore(predictor, defaultScore)
        testX = testg.getX()

        cqr = SCC(calg, predictor, qr, alpha)
        cs, isinf = cqr.predict(testX, defaultSolveScore)
        size = cs[:, 1] - cs[:, 0]
        cov = [np.maximum(coverage(cs[i, :], testX[i, :], g), isinf[i]) for i in range(cs.shape[0])]
        print(f"CQR: {np.mean(size):.4f}, {np.mean(cov):.4f}, {glcp.q:.4f}")
        lb[i, :] = np.mean(size), np.mean(cov)
    print(np.mean(lb, axis=0))
