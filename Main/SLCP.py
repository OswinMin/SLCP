import numpy as np
from tools import *
from Agents import *
from Predictor import *
from Tuner import *
from QuanRegressor import *

class SLCP:
    def __init__(self, targetAgent:Agent, X:np.ndarray, generator:Generator, predictor:Predictor, tune_layers:list[int]):
        """
        :param targetAgent:
        :param X:   unlabeled covariates
        :param eng:
        :param predictor:
        """
        self.targetAgent = targetAgent
        self.X = X
        self.generator = deepcopy(generator)
        self.predictor = predictor
        self.tune_layers = tune_layers
        self.calN = self.targetAgent.n
        self.generate_n = 500
        self.tuner = Tuner(self.generator, tune_layers)

    def tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd:float=1., temperature=10., **kwargs):
        # self.generator.loadMax(0.005, np.max(self.targetAgent.getS()))
        self.tuner.tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, lbd, temperature)
        self.beta = self.generator.cdf(self.X, self.tuner.generaten(self.X, n), self.generate_n).reshape(-1)
        self.q = np.quantile(self.beta, (1 - alpha)*(self.calN+1)/self.calN, method='higher')

    def auto_lbd_tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd_list=(1.,), temperature=10., gap=1e-2, **kwargs):
        active_lbd = []
        level = (1 - alpha)*(self.calN+1)/self.calN
        upper, lower = np.quantile(self.targetAgent.getS(), level+gap), np.quantile(self.targetAgent.getS(), level-gap)
        for lbd in lbd_list:
            tuner = deepcopy(self.tuner)
            tuner.tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, lbd, temperature)

    def predict(self, testX: np.ndarray, solveScore=defaultSolveScore):
        sq = self.generator.quantile(testX, self.q, self.generate_n)
        cs = solveScore(testX, sq, self.predictor)
        return cs

class SLCP_SCC:
    def __init__(self, targetAgent:Agent, X:np.ndarray, generator:Generator, qrmodel:QRModel, predictor:Predictor, tune_layers:list[int]):
        """
        :param targetAgent:
        :param X:   unlabeled covariates
        :param eng:
        :param predictor:
        """
        self.targetAgent = targetAgent
        self.X = X
        self.generator = deepcopy(generator)
        self.qrmodel = deepcopy(qrmodel)
        self.predictor = predictor
        self.tune_layers = tune_layers
        self.calN = self.targetAgent.n
        self.generate_n = 500
        self.tuner = Tuner(self.generator, tune_layers)

    def tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd:float=1., temperature=10., **kwargs):
        # self.generator.loadMax(0.005, np.max(self.targetAgent.getS()))
        self.tuner.tune_marginal_scc(self.targetAgent.getX(), self.targetAgent.getS(), self.X, self.qrmodel, n, epochs, learning_rate, n_grid, lbd, temperature)
        self.beta = (self.tuner.generaten(self.X, n) - self.qrmodel.predict(self.X).reshape((-1,1))).reshape(-1)
        self.qhat = np.quantile(self.beta, (1 - alpha)*(self.calN+1)/self.calN, method='higher')

    def predict(self, testX: np.ndarray, solveScore=defaultSolveScore):
        sq = np.maximum(self.qrmodel.predict(testX).reshape(-1) + self.qhat, 0)
        cs = solveScore(testX, sq, self.predictor)
        return cs

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
    predTrAgents = [Agent(*generateXY(ga, n // 2, d)) for ga in gamma]
    calTrAgents = [Agent(*generateXY(ga, n // 2, d)) for ga in gamma]
    predictorList = [Predictor('lr') for _ in predTrAgents]
    for i in range(len(predictorList)):
        predictorList[i].trainFromAgent(predTrAgents[i])
        calTrAgents[i].calScore(predictorList[i], defaultScore)
    calTrAgent = combineAgents(calTrAgents)
    generator = Generator(d, [20, 50, 20], False, d)
    generator.trainEng(calTrAgent.getX(), calTrAgent.getS(), 10, 32, 300, 0.005, mute=True)

    alpha, g, reps = 0.1, 0.9, 10
    lb = np.zeros((reps, 2))
    for i in range(reps):
        setseed(i + 1)
        trg = Agent(*generateXY(g, 30, d))
        calg = Agent(*generateXY(g, 30, d))
        testg = Agent(*generateXY(g, 100, d))
        predictor = Predictor('lr')
        predictor.trainFromAgent(trg)
        calg.calScore(predictor, defaultScore), testg.calScore(predictor, defaultScore)
        testX = testg.getX()
        semig = Agent(*generateXY(g, 1000, d))

        slcp = SLCP(calg, semig.getX(), generator, predictor, [2])
        slcp.tune(2, 300, 5e-3, alpha, 50, 0.01, 10.)
        cs = slcp.predict(testX, defaultSolveScore)
        size = cs[:, 1] - cs[:, 0]
        cov = [coverage(cs[i, :], testX[i, :], g) for i in range(cs.shape[0])]
        print(f"SLCP: {np.mean(size):.4f}, {np.mean(cov):.4f}, {slcp.q:.4f}")
        lb[i, :] = np.mean(size), np.mean(cov)
    print(np.mean(lb, axis=0))
