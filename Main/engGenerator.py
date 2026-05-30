import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tools import *

def CvM(u:np.ndarray):
    n = len(u)
    u_sorted = np.sort(u)
    i = np.arange(1, n + 1)
    cvm_statistic = (1 / (12 * n)) + np.sum((u_sorted - (2 * i - 1) / (2 * n)) ** 2)
    return cvm_statistic

def Wasserstein(u:np.ndarray):
    u_sorted = np.sort(u)
    n = len(u)
    wass = np.mean(np.abs(u_sorted - (np.arange(n) + 0.5) / n))
    return wass

class Generator(nn.Module):
    def __init__(self, inputDim:int, hiddenDim:list[int], randNum:int=1, **kwargs):
        """
        :param inputDim: int
        :param hiddenDim: list of int
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
        self.noise_scalar = 1.

    def forward(self, xep:torch.Tensor):
        """
        :param xep: n*(inputDim+randNum)
        :return: n*1
        """
        xep = self.model(xep)
        return xep

    def predict(self, xep:np.ndarray):
        """
        :param xep: n*(inputDim+randNum)
        :return: n*1 np.ndarray
        """
        return self.forward(torch.tensor(xep).float()).detach().numpy()

    def generaten(self, x:np.ndarray, n:int=1, seed=0):
        """
        :param x: d or 1*d or m*d
        :param n:
        :return: m * n
        """
        x = x.reshape((-1, self.inputDim))
        m = x.shape[0]
        original_state = np.random.get_state()
        try:
            np.random.seed(seed)
            ep = np.random.normal(0, 1, (n*x.shape[0],self.randNum)) * self.noise_scalar
        finally:
            np.random.set_state(original_state)
        xep = np.concatenate([x.repeat(n, 0), ep], axis=1)
        return self.predict(xep).reshape((m, n))

    def trainEng(self, X:np.ndarray, Y:np.ndarray, m:int=100, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, isLog=False, path='', log=None, mute=True, **kwargs):
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
        # self.cal_scalar(X, Y, m, kwargs.get('seed', 0), stat_type=kwargs.get('stat_type', 'CvM'))

    def __inner_train(self, dataloader, optimizer, epochs, m, isLog=False, path='', log=None, mute=True, **kwargs):
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

    def cal_scalar(self, X:np.ndarray, Y:np.ndarray, m:int=100, seed=0, stat_type='CvM'):
        stat_fun = CvM if stat_type == 'CvM' else Wasserstein
        def fun(scalar):
            self.noise_scalar = scalar[0]
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                marginal_cdf = self.cdf(X, Y, m).reshape(-1)
            finally:
                np.random.set_state(state)
            return stat_fun(marginal_cdf)
        from scipy.optimize import minimize
        res = minimize(fun, np.array([1.0]), bounds=[(0.1, 10)], method='Nelder-Mead')
        self.noise_scalar = res.x[0]

    def cdf(self, X:np.ndarray, Y:np.ndarray, m:int=100, **kwargs):
        """
        :param X: n * inputDim
        :param Y: n * 1 or n * k
        :param m: number of epsilons generated for each X
        :return: P(Y<=y|X=x) with shape same as Y
        """
        ydim = len(Y.shape)
        X = X.reshape((-1, self.inputDim))
        Y = Y.reshape((-1, 1)) if len(Y.shape) == 1 else Y              # n * 1
        preds = self.generaten(X, m)                                    # n * m
        cdf = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            cdf[:, i] = (preds <= Y[:, i].reshape((-1, 1))).mean(-1)    # n * 1
        return cdf.reshape(-1) if ydim == 1 else cdf

    def quantile(self, X:np.ndarray, q:float=.9, m:int=100, **kwargs):
        """
        :param X: n * inputDim
        :param q: float
        :param m: number of epsilons generated for each X
        :return: Q(q|X=x) of shape (n,)
        """
        X = X.reshape((-1, self.inputDim))
        preds = self.generaten(X, m)    # n * m
        qhat = np.quantile(preds, q, axis=-1, method='higher')
        return qhat.reshape(-1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.stats as ss
    np.random.seed(1)
    torch.manual_seed(1)
    X = np.random.uniform(-4, 4, (50, 5))
    Y = X.sum(-1) + np.cos(X[:, 0]) * np.random.normal(0, 1, 50)

    np.random.seed(1)
    torch.manual_seed(1)
    generator = Generator(5, [20, 50, 20], randNum=5)
    generator.trainEng(X.reshape((-1, 5)), Y.reshape((-1, 1)), m=10, batch_size=32, epochs=100, log=log, mute=False, seed=10, stat_type='CvM')

    cdf = generator.cdf(X, Y, 100)
    print(CvM(cdf), generator.noise_scalar)
