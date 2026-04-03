import numpy as np
from QuanRegressor import *
from engGenerator import *
from tools import *
from copy import deepcopy
import torch.nn.functional as F

class Tuner(nn.Module):
    def __init__(self, generator:Generator, tune_layers: list[int]):
        super(Tuner, self).__init__()
        self.original_generator = deepcopy(generator)
        self.tune_layers = list(np.sort(tune_layers))
        self.adjusted_tune_layers = []
        for param in self.original_generator.parameters():
            param.requires_grad = False
        self.delta_weights = nn.ParameterList()
        self.delta_biases = nn.ParameterList()

        idx = 0
        for i, layer in enumerate(self.original_generator.layers):
            if isinstance(layer, nn.Linear):
                if idx in tune_layers:
                    self.adjusted_tune_layers.append(i)
                idx += 1
        for i, layer in enumerate(self.original_generator.layers):
            if isinstance(layer, nn.Linear) and i in self.adjusted_tune_layers:
                # 初始化ΔW和Δb为零
                delta_w = nn.Parameter(torch.zeros_like(layer.weight))
                delta_b = nn.Parameter(torch.zeros_like(layer.bias))
                self.delta_weights.append(delta_w)
                self.delta_biases.append(delta_b)
            else:
                # 对于不需要微调的层，添加空的占位符
                self.delta_weights.append(None)
                self.delta_biases.append(None)

        self.inputDim = self.original_generator.inputDim
        self.randNum = self.original_generator.randNum
        self.layer_sizes = self.original_generator.layer_sizes
        self.S = nn.Sigmoid()
        self.restrict = self.original_generator.restrict
        self.maxq = self.original_generator.maxq
        self.maxS = self.original_generator.maxS

    def forward(self, xep: torch.Tensor):
        """
        前向传播：使用 W + ΔW 进行计算
        """
        x = xep
        for i, layer in enumerate(self.original_generator.layers):
            if isinstance(layer, nn.Linear):
                if i in self.adjusted_tune_layers:
                    # 使用 W + ΔW, b + Δb
                    weight = layer.weight + self.delta_weights[i]
                    bias = layer.bias + self.delta_biases[i]
                    x = F.linear(x, weight, bias)
                else:
                    x = layer(x)    # 使用原始参数
            else:
                x = layer(x)    # 激活函数层
        if self.original_generator.restrict:
            x = self.original_generator.S(x)
        return x

    def get_delta_norm(self):
        """计算所有ΔW参数的范数，用于正则化"""
        total_norm = 0.0
        for delta_w in self.delta_weights:
            if delta_w is not None:
                total_norm += (delta_w ** 2).sum()
        for delta_b in self.delta_biases:
            if delta_b is not None:
                total_norm += (delta_b ** 2).sum()
        return total_norm

    def predict(self, xep:np.ndarray):
        """
        :param xep: n*inputDim+1
        :return: n*1 np.ndarray
        """
        return self.forward(torch.tensor(xep).float()).detach().numpy()

    def generaten(self, x:np.ndarray, n:int=1):
        """
        :param x: d or 1*d or m*d
        :param n:
        :return: m * n
        """
        if len(x.shape) == 1:
            x = x.reshape((-1, self.inputDim))
        m = x.shape[0]
        ep = np.random.normal(0, 1, (n*x.shape[0],self.randNum))
        xep = np.concatenate([x.repeat(n, 0), ep], axis=1)
        return self.predict(xep).reshape((m, n))

    def _generaten(self, x:np.ndarray, n:int=1):
        if len(x.shape) == 1:
            x = x.reshape((-1, self.inputDim))
        ep = np.random.normal(0, 1, (n*x.shape[0],self.randNum))
        xep = np.concatenate([x.repeat(n, 0), ep], axis=1)
        return self.forward(torch.tensor(xep).float()).view((-1, n))

    def cdf(self, X:np.ndarray, Y:np.ndarray, m:int=100):
        """
        :param X: n * inputDim
        :param Y: n * 1 or n * k
        :param m: number of epsilons generated for each X
        :return: P(Y<=y|X=x) with shape same as Y
        """
        ydim = len(Y.shape)
        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))
        if len(X.shape) == 1:
            X = X.reshape((-1, self.inputDim))
        w = np.random.uniform(0, self.maxq, (X.shape[0], 1))    # n * 1
        preds = self.generaten(X, m)    # n * m
        weight = np.ones_like(preds)    # n * m
        weight = weight / m * (1 - w)   # n * m
        # n * (m+1)
        preds = np.concatenate((preds, np.ones((X.shape[0],1))*self.maxS), axis=1)
        # n * (m+1)
        weight = np.concatenate((weight, w), axis=1)
        cdf = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            cdf[:, i] = (weight * (preds <= Y[:, [i]])).sum(-1)
        if ydim == 1:
            return cdf.reshape(-1)
        else:
            return cdf

    def quantile(self, X:np.ndarray, q:float=.9, m:int=100):
        """
        :param X: n * inputDim
        :param q: float
        :param m: number of epsilons generated for each X
        :return: Q(q|X=x) of shape (n,)
        """
        if len(X.shape) == 1:
            X = X.reshape((-1, self.inputDim))
        w = np.random.uniform(0, self.maxq, (X.shape[0], 1))    # n * 1
        preds = self.generaten(X, m)    # n * m
        weight = np.ones_like(preds)    # n * m
        weight = weight / m * (1 - w)   # n * m
        # n * (m+1)
        preds = np.concatenate((preds, np.ones((X.shape[0], 1)) * self.maxS), axis=1)
        # n * (m+1)
        weight = np.concatenate((weight, w), axis=1)
        qhat = empiricalQuantile(preds, weight, q)
        return qhat.reshape(-1)

    def tune_marginal(self, labeledX:np.ndarray, labeledY:np.ndarray, unlabeledX:np.ndarray, n:int=10, epochs:int=100, learning_rate:float=0.01, n_grid=50, lbd:float=1., temperature=10.):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        nmin = min(labeledX.shape[0], unlabeledX.shape[0]*n)
        q = torch.tensor(np.linspace(1/nmin, 1-1/nmin, n_grid)).float()

        empiricalY = torch.tensor(labeledY).view(-1).float()
        orig_empiricalY = torch.tensor(self.original_generator.generaten(labeledX, 200)).float()
        empiricalBeta = torch.sigmoid((empiricalY.view((-1, 1)).unsqueeze(2) - orig_empiricalY.unsqueeze(1)) * temperature).mean(dim=2)
        empiricalQ = torch.quantile(empiricalBeta, q)

        orig_generatedY = torch.tensor(self.original_generator.generaten(unlabeledX, 200)).float()
        for ep in range(epochs):
            generatedY = self._generaten(unlabeledX, n)
            generatedBeta = torch.sigmoid((generatedY.unsqueeze(2) - orig_generatedY.unsqueeze(1)) * temperature).mean(dim=2)
            generatedQ = torch.quantile(generatedBeta, q)
            L = torch.mean(torch.abs(empiricalQ - generatedQ)) + lbd*self.get_delta_norm()
            # L = torch.mean(torch.abs(empiricalQ - generatedQ))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

    def tune_marginal_scc(self, labeledX: np.ndarray, labeledY: np.ndarray, unlabeledX: np.ndarray, qrmodel:QRModel, n: int = 10, epochs: int = 100, learning_rate: float = 0.01, n_grid=50, lbd: float = 1., temperature=10.):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        nmin = min(labeledX.shape[0], unlabeledX.shape[0]*n)
        q = torch.tensor(np.linspace(1/nmin, 1-1/nmin, n_grid)).float()

        empiricalY = torch.tensor(labeledY).view(-1).float()
        empiricalBeta = (empiricalY - torch.tensor(qrmodel.predict(labeledX)).view(-1).float()).view(-1, 1)
        empiricalQ = torch.quantile(empiricalBeta, q)

        orig_generatedY = torch.tensor(qrmodel.predict(unlabeledX)).view(-1, 1).float()
        for ep in range(epochs):
            generatedY = self._generaten(unlabeledX, n)
            generatedBeta = generatedY - orig_generatedY
            generatedQ = torch.quantile(generatedBeta, q)
            L = torch.mean(torch.abs(empiricalQ - generatedQ)) + lbd*self.get_delta_norm()
            # L = torch.mean(torch.abs(empiricalQ - generatedQ))
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

if __name__ == '__main__':
    import scipy.stats as ss
    import matplotlib.pyplot as plt

    np.random.seed(1)
    torch.manual_seed(1)
    X = np.random.uniform(-4, 4, 200)
    Y = X + np.cos(X) * np.random.normal(0, 1, 200)
    generator = Generator(1, [20, 50, 20], restrict=False, randNum=3)
    generator.trainEng(X.reshape((-1, 1)), Y.reshape((-1, 1)), m=10, batch_size=32, epochs=300, log=log, mute=False)

    tuner = Tuner(generator, [1])
    X_ = np.repeat(X, 5, axis=0)
    yhat = tuner.generaten(X.reshape((-1, 1)), 5)

    # plt.scatter(X, Y, s=1)
    # plt.show()
    # plt.scatter(X_, yhat, s=1)
    # plt.show()
    #
    # cdf = generator.cdf(X, Y, 100)
    # trueCDF = ss.norm.cdf((Y - X) / np.abs(np.cos(X)))
    # print(f"{np.abs(cdf - trueCDF).mean():.4f}")
    #
    # quantiles = generator.quantile(X, 0.9, 100)
    # cdfQ = ss.norm.cdf((quantiles - X) / np.abs(np.cos(X)))
    # print(f"{np.abs(cdfQ - 0.9).mean():.4f}")

    setseed(5)
    tuner = Tuner(generator, [1])
    labeledX = X[:50]
    labeledY = labeledX + 1.2 * np.cos(labeledX) * np.random.normal(0, 1, 50)
    unlabeledX = np.random.uniform(-4, 4, 1000)

    print(np.quantile(tuner.cdf(labeledX, labeledY, 200), 0.9))
    generatedY = tuner.generaten(unlabeledX, 5)
    print(np.quantile(tuner.cdf(unlabeledX, generatedY, 200), 0.9))
    tuner.tune_marginal(labeledX, labeledY, unlabeledX, n=1, learning_rate=0.005)
    generatedY_ = tuner.generaten(unlabeledX, 5)
    print(np.quantile(tuner.cdf(unlabeledX, generatedY_, 200), 0.9))
