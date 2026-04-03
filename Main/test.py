import numpy as np
from GLCP import *
from SLCP import *
import scipy.stats as ss
from tools import *
from Agents import *
from engGenerator import *

def generateXY(gamma, n, d):
    X = np.random.normal(0, 1, (n, d))
    Y = np.random.normal(0, 1, (n, 1)) * ((np.abs(X).sum(-1)) ** 0.5).reshape((-1, 1)) * gamma
    return d, n, X, Y
def coverage(interval, x, gamma):
    interval = [num / ((np.abs(x).sum() ** 0.5) * gamma) for num in interval]
    return ss.norm.cdf(interval[1]) - ss.norm.cdf(interval[0])

setseed(0)
n = 200
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
lb, sm = np.zeros((reps, 2)), np.zeros((reps, 2))
s0 = np.zeros(reps)
for i in range(reps):
    setseed(i+1)
    trg = Agent(*generateXY(g, 20, d))
    calg = Agent(*generateXY(g, 20, d))
    testg = Agent(*generateXY(g, 100, d))
    predictor = Predictor('lr')
    predictor.trainFromAgent(trg)
    calg.calScore(predictor, defaultScore), testg.calScore(predictor, defaultScore)
    testX = testg.getX()
    semig = Agent(*generateXY(g, 1000, d))

    s0[i] = np.quantile(calg.getS().reshape(-1), (1-alpha)*(20+1)/20, method='higher') * 2

    setseed(i+1)
    glcp = GLCP(calg, generator, predictor)
    cs = glcp.predict(testX, 500, alpha, defaultSolveScore)
    size = cs[:, 1] - cs[:, 0]
    cov = [coverage(cs[i, :], testX[i, :], g) for i in range(cs.shape[0])]
    print(f"DCP: {np.mean(size):.4f}, {np.mean(cov):.4f}, {glcp.q:.4f}")
    lb[i, :] = np.mean(size), np.mean(cov)

    setseed(i+1)
    slcp = SLCP(calg, semig.getX(), generator, predictor, [2])
    slcp.tune(2, 200, 5e-3, alpha, 50, 0.008, 10.)
    cs = slcp.predict(testX, defaultSolveScore)
    size = cs[:, 1] - cs[:, 0]
    cov = [coverage(cs[i, :], testX[i, :], g) for i in range(cs.shape[0])]
    print(f"SLCP: {np.mean(size):.4f}, {np.mean(cov):.4f}, {slcp.q:.4f}")
    sm[i, :] = np.mean(size), np.mean(cov)
print(np.mean(lb, axis=0), np.std(lb, axis=0))
print(np.mean(sm, axis=0), np.std(sm, axis=0))
print(np.mean(s0, axis=0), np.std(s0, axis=0))
