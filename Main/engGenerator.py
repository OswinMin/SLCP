import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tools import *

class Generator(nn.Module):
    def __init__(self, inputDim:int, hiddenDim:list[int], restrict:bool=False, randNum:int=1, maxq:float=0.005, maxS=20., **kwargs):
        """
        :param inputDim: int
        :param hiddenDim: list of int
        :param restrict: if restrict output in [0,1]
        :param randNum: number of epsilon to control
        """
        super(Generator, self).__init__()
        self.inputDim = inputDim
        self.randNum = randNum
        self.layer_sizes = [inputDim+randNum] + hiddenDim + [1]
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:
                self.layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*self.layers)
        self.S = nn.Sigmoid()
        self.restrict = restrict
        self.maxq = maxq
        self.maxS = maxS

    def loadMax(self, maxq:float=0.005, maxS=20.):
        self.maxq = maxq
        self.maxS = maxS

    def forward(self, xep:torch.Tensor):
        """
        :param xep: n*inputDim+1
        :return: n*1
        """
        xep = self.model(xep)
        if self.restrict:
            xep = self.S(xep)
        return xep

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

    def trainEng(self, X:np.ndarray, Y:np.ndarray, m:int=100, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, isLog=False, path='', log=None, mute=True, expand_kwargs=None):
        """
        Use X, Y train a simple predictor
        :param X: n*inputDim
        :param Y: n*1
        :param m: number of epsilons generated for each X
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :return: No return
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.__inner_train(dataloader, optimizer, epochs, m, isLog, path, log, mute)
        if expand_kwargs is not None:
            # ratio : number of sythetic samples to expand
            # sided : 0 only lower; 1 only upper; 2 two sided
            ratio, sided, expand_ratio = expand_kwargs['ratio'], expand_kwargs['sided'], expand_kwargs['expand_ratio']
            ratioN = int(X.shape[0] * ratio)
            if sided == 2:
                ratioN = ratioN // 2
            idx = np.random.choice(X.shape[0], size=ratioN, replace=True)
            X_ = X[idx]     # ratioN * inputDim
            Y_ = self.generaten(X_, m)      # ratioN * m
            Y_range = np.concatenate([np.max(Y_, axis=1, keepdims=True), np.min(Y_, axis=1, keepdims=True)], axis=1)
            Y_up = (Y_range[:, 0] + expand_ratio * (Y_range[:, 0] - Y_range[:, 1])).reshape((-1,1))
            Y_low = (Y_range[:, 1] - expand_ratio * (Y_range[:, 0] - Y_range[:, 1])).reshape((-1,1))
            if sided != 2:
                X = np.concatenate([X, X_], axis=0)
                if sided == 0:
                    Y = np.concatenate([Y, Y_low], axis=0)
                else:
                    Y = np.concatenate([Y, Y_up], axis=0)
            else:
                X = np.concatenate([X, X_, X_], axis=0)
                Y = np.concatenate([Y, Y_low, Y_up], axis=0)
            self.trainEng(X, Y, m, batch_size, epochs, learning_rate, isLog, path, log, mute, expand_kwargs=None)

    def __inner_train(self, dataloader, optimizer, epochs, m, isLog=False, path='', log=None, mute=True):
        for epoch in range(epochs):
            ttloss = 0
            ttloss1 = 0
            ttgain1 = 0
            t = 0
            for inputs, targets in dataloader:
                bs = inputs.shape[0]
                inputs = inputs.repeat_interleave(m, 0)
                ep = torch.tensor(np.random.normal(0, 1, (bs*m, self.randNum))).float()
                inputs = torch.concat([inputs, ep], axis=1)
                outputs = self(inputs)
                loss1 = torch.mean(torch.abs(outputs-targets.repeat_interleave(m, 0)))
                outputs = outputs.view(bs, m)
                gain1 = torch.sum(torch.abs(outputs[:,:,None]-outputs[:,None,:])) / (2*bs*m*(m-1))
                loss = loss1 - gain1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ttloss += loss.item()
                ttloss1 += loss1.item()
                ttgain1 += gain1.item()
                t += 1
            if not mute:
                if epoch % (epochs//10) == max((epochs//10) - 1,1):
                    log(f'Epoch [{epoch + 1}/{epochs}], Loss: {ttloss/t:.4f}, Error: {ttloss1/t:.4f}, Gain: {ttgain1/t*2:.4f}', path, isLog)

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

    def quantile(self, X:np.ndarray, q:float=.9, m:int=100, **kwargs):
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.stats as ss
    np.random.seed(1)
    torch.manual_seed(1)
    X = np.random.uniform(-4, 4, 2000)
    Y = X + np.cos(X) * np.random.normal(0, 1, 2000)
    plt.scatter(X, Y, s=1)
    plt.show()

    np.random.seed(1)
    torch.manual_seed(1)
    X = np.random.uniform(-4, 4, 200)
    Y = X + np.cos(X) * np.random.normal(0, 1, 200)

    np.random.seed(1)
    torch.manual_seed(1)
    generator = Generator(1, [20, 50, 20], restrict=False, randNum=3)
    generator.trainEng(X.reshape((-1, 1)), Y.reshape((-1, 1)), m=10, batch_size=32, epochs=600, log=log, mute=False)
    X_ = np.repeat(X, 5, axis=0)
    yhat = generator.generaten(X.reshape((-1, 1)), 5)
    plt.scatter(X_, yhat, s=1)
    plt.show()

    np.random.seed(1)
    torch.manual_seed(1)
    generator = Generator(1, [20, 50, 20], restrict=False, randNum=3)
    generator.trainEng(X.reshape((-1,1)), Y.reshape((-1,1)), m=10, batch_size=32, epochs=300, log=log, mute=False, expand_kwargs={'ratio':0.1, 'sided':2, 'expand_ratio':0.1})
    X_ = np.repeat(X, 5, axis=0)
    yhat = generator.generaten(X.reshape((-1,1)), 5)
    plt.scatter(X_, yhat, s=1)
    plt.show()

    cdf = generator.cdf(X, Y, 100)
    trueCDF = ss.norm.cdf((Y-X)/np.abs(np.cos(X)))
    print(f"{np.abs(cdf-trueCDF).mean():.4f}")

    quantiles = generator.quantile(X, 0.9, 100)
    cdfQ = ss.norm.cdf((quantiles-X)/np.abs(np.cos(X)))
    print(f"{np.abs(cdfQ-0.9).mean():.4f}")
