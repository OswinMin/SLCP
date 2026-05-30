import numpy as np
from tools import *
from Agents import *
from Predictor import *
from Tuner import *
from QuanRegressor import *

def check_order(part1, delta):
    """
    delta 保证递减，否则前面较小lambda换给后面较大的
    在 delta 递减时，part1递增，否则把后面较小的给前面较大的
    return: idx1, idx2，最终要把 idx1 对应参数给 idx2 继续训练
    """
    for idx1 in range(1, len(delta)):
        for idx2 in range(idx1+1, len(delta)):
            if (delta[idx2] < delta[idx1]) and (part1[idx2] < part1[idx1]):
                return idx2, idx1
            if (delta[idx2] >= delta[idx1]) and (part1[idx2] >= part1[idx1]):
                return idx1, idx2
    return None, None

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

    def load_tuner(self, tuner:Tuner, copy=True, **kwargs):
        self.tuner = deepcopy(tuner) if copy else tuner
        self.beta = self.tuner.generate_beta(self.X, kwargs.get('m', 200), kwargs.get('n', 10), kwargs.get('temperature', 10.)).reshape(-1)
        self.q = np.quantile(self.beta, (1 - kwargs.get('alpha', .1)) * (self.calN + 1) / self.calN, method='higher')

    def tune_lbd_list(self, n=10, epochs:int=100, learning_rate=5e-3, n_grid=50, lbd_list=(0.,), temperature=10., **kwargs):
        lbd_list = list(lbd_list) if 0. in lbd_list else [0.] + list(lbd_list)
        lbd_list = sorted(lbd_list)
        tuner_list, part1, delta = [None] * len(lbd_list), [None] * len(lbd_list), [None] * len(lbd_list)
        part1[0], delta[0] = self.tuner.tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, 0., temperature, **kwargs)
        tuner_list[0] = deepcopy(self.tuner)
        for i in range(1, len(lbd_list)):
            tuner = deepcopy(self.tuner)
            part1[i], delta[i] = tuner.tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, lbd_list[i], temperature, **kwargs)
            tuner_list[i] = deepcopy(tuner)
        idx1, idx2 = check_order(part1, delta)
        tot = 1
        while (idx1 is not None) and (tot < 3 * len(lbd_list)):
            tot += 1
            tuner_list[idx2] = deepcopy(tuner_list[idx1])
            part1[idx2], delta[idx2] = tuner_list[idx2].tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, lbd_list[idx2], temperature, **kwargs)
            idx1, idx2 = check_order(part1, delta)
        return tuner_list

    def tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd:float=1., temperature=10., **kwargs):
        self.tuner.tune_marginal(self.targetAgent.getX(), self.targetAgent.getS(), self.X, n, epochs, learning_rate, n_grid, lbd, temperature, **kwargs)
        self.beta = self.tuner.generate_beta(self.X, kwargs.get('m', 200), n, temperature).reshape(-1)
        self.q = np.quantile(self.beta, (1 - alpha)*(self.calN+1)/self.calN, method='higher')

    def auto_lbd_tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd_list=(1.,), temperature=10., gap=1e-2, **kwargs):
        if kwargs.get('tuner_list', None) is None:
            tuner_list = self.tune_lbd_list(n, epochs, learning_rate, n_grid, lbd_list, temperature, **kwargs)
        else:
            tuner_list = kwargs.get('tuner_list')
        beta = self.generator.cdf(self.targetAgent.getX(), self.targetAgent.getS(), kwargs.get('m', 200)).reshape(-1)
        lower, up = np.quantile(beta, (1 - alpha)*(self.calN+1)/self.calN-gap, method='lower'), np.quantile(beta, (1 - alpha)*(self.calN+1)/self.calN+gap, method='higher')
        q_list, acitive_idx = [], []
        for idx, tuner in enumerate(tuner_list):
            q = np.quantile(tuner.generate_beta(self.X, kwargs.get('m', 200), n, temperature).reshape(-1), (1 - alpha)*(self.calN+1)/self.calN, method='higher')
            q_list.append(q)
            if lower <= q <= up:
                acitive_idx.append(idx)
        if len(acitive_idx) == 0:
            idx = 0
        else:
            idx = acitive_idx[np.argmax(np.array(lbd_list)[acitive_idx])]
        self.q = q_list[idx]
        self.tuner = tuner_list[idx]
        return tuner_list, q_list, idx

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

    def load_tuner(self, tuner:Tuner, copy=True, **kwargs):
        self.tuner = deepcopy(tuner) if copy else tuner
        self.beta = self.tuner.generate_beta_scc(self.X, self.qrmodel, kwargs.get('n', 10)).reshape(-1)
        self.qhat = np.quantile(self.beta, (1 - kwargs.get('alpha', .1)) * (self.calN + 1) / self.calN, method='higher')

    def tune_lbd_list(self, n=10, epochs:int=100, learning_rate=5e-3, n_grid=50, lbd_list=(0.,), **kwargs):
        lbd_list = list(lbd_list) if 0. in lbd_list else [0.] + list(lbd_list)
        lbd_list = sorted(lbd_list)
        tuner_list, part1, delta = [None] * len(lbd_list), [None] * len(lbd_list), [None] * len(lbd_list)
        part1[0], delta[0] = self.tuner.tune_marginal_scc(self.targetAgent.getX(), self.targetAgent.getS(), self.X, self.qrmodel, n, epochs, learning_rate, n_grid, 0., **kwargs)
        tuner_list[0] = deepcopy(self.tuner)
        for i in range(1, len(lbd_list)):
            tuner = deepcopy(self.tuner)
            part1[i], delta[i] = tuner.tune_marginal_scc(self.targetAgent.getX(), self.targetAgent.getS(), self.X, self.qrmodel, n, epochs, learning_rate, n_grid, lbd_list[i], **kwargs)
            tuner_list[i] = deepcopy(tuner)
        idx1, idx2 = check_order(part1, delta)
        tot = 1
        while (idx1 is not None) and (tot < 3*len(lbd_list)):
            tot += 1
            tuner_list[idx2] = deepcopy(tuner_list[idx1])
            part1[idx2], delta[idx2] = tuner_list[idx2].tune_marginal_scc(self.targetAgent.getX(), self.targetAgent.getS(), self.X, self.qrmodel, n, epochs, learning_rate, n_grid, lbd_list[idx2], **kwargs)
            idx1, idx2 = check_order(part1, delta)
        return tuner_list

    def tune(self, n=10, epochs:int=100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd:float=1., **kwargs):
        self.tuner.tune_marginal_scc(self.targetAgent.getX(), self.targetAgent.getS(), self.X, self.qrmodel, n, epochs, learning_rate, n_grid, lbd, **kwargs)
        self.beta = self.tuner.generate_beta_scc(self.X, self.qrmodel, n).reshape(-1)
        self.qhat = np.quantile(self.beta, (1 - alpha)*(self.calN+1)/self.calN, method='higher')

    def auto_lbd_tune(self, n=10, epochs: int = 100, learning_rate=5e-3, alpha: float = 0.1, n_grid=50, lbd_list=(1.,), gap=1e-2, **kwargs):
        if kwargs.get('tuner_list', None) is None:
            tuner_list = self.tune_lbd_list(n, epochs, learning_rate, n_grid, lbd_list, **kwargs)
        else:
            tuner_list = kwargs.get('tuner_list')
        beta = self.targetAgent.getS().reshape(-1) - self.qrmodel.predict(self.targetAgent.getX()).reshape(-1)
        lower, up = np.quantile(beta, (1 - alpha)*(self.calN+1)/self.calN-gap, method='lower'), np.quantile(beta, (1 - alpha)*(self.calN+1)/self.calN+gap, method='higher')
        qhat_list, acitive_idx = [], []
        for idx, tuner in enumerate(tuner_list):
            qhat = np.quantile(tuner.generate_beta_scc(self.X, self.qrmodel, n).reshape(-1), (1 - alpha)*(self.calN+1)/self.calN, method='higher')
            qhat_list.append(qhat)
            if lower <= qhat <= up:
                acitive_idx.append(idx)
        if len(acitive_idx) == 0:
            idx = 0
        else:
            idx = acitive_idx[np.argmax(np.array(lbd_list)[acitive_idx])]
        self.qhat = qhat_list[idx]
        self.tuner = tuner_list[idx]
        return tuner_list, qhat_list, idx

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
