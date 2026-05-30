import numpy as np
from QuanRegressor import *
from engGenerator import *
from tools import *
from copy import deepcopy
import torch.nn.functional as F
from torch.optim import lr_scheduler

penalty_fun = {
    'MAE': lambda x1, x2: torch.mean(torch.abs(x1 - x2)),
    'MSE': lambda x1, x2: torch.mean((x1 - x2)**2),
    'MaxAE': lambda x1, x2: torch.max(torch.abs(x1 - x2)),
    'MaxSAE': lambda x1, x2: torch.max(torch.abs(x1 - x2))**2,
}

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
            if isinstance(layer, nn.Linear) and (i in self.adjusted_tune_layers):
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
        self.noise_scalar = self.original_generator.noise_scalar
        self.layer_sizes = self.original_generator.layer_sizes

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

    def generaten(self, x:np.ndarray, n:int=1, seed=0):
        """
        :param x: d or 1*d or m*d
        :param n:
        :return: m * n  np.ndarray
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

    def _generaten(self, x:np.ndarray, n:int=1, seed=0):
        x = x.reshape((-1, self.inputDim))
        original_state = np.random.get_state()
        try:
            np.random.seed(seed)
            ep = np.random.normal(0, 1, (n*x.shape[0],self.randNum)) * self.noise_scalar
        finally:
            np.random.set_state(original_state)
        xep = np.concatenate([x.repeat(n, 0), ep], axis=1)
        return self.forward(torch.tensor(xep).float()).view((-1, n))    # tensor

    def cdf(self, X:np.ndarray, Y:np.ndarray, m:int=100):
        """
        :param X: n * inputDim
        :param Y: n * 1 or n * k
        :param m: number of epsilons generated for each X
        :return: P(Y<=y|X=x) with shape same as Y
        """
        ydim = len(Y.shape)
        Y = Y.reshape((-1, 1)) if len(Y.shape) == 1 else Y
        X = X.reshape((-1, self.inputDim))
        preds = self.generaten(X, m)    # n * m
        cdf = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            cdf[:, i] = (preds <= Y[:, [i]]).mean(-1)
        return cdf.reshape(-1) if ydim == 1 else cdf

    def quantile(self, X:np.ndarray, q:float=.9, m:int=100):
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

    def generate_beta(self, unlabeledX:np.ndarray, m:int, n:int=10, temperature=10.):
        """
        :param unlabeledX: N * d
        :param m:
        :param n:
        :param temperature:
        :return: N * n, np.ndarray, each element is the generated beta for one generated Y
        """
        orig_generatedY = torch.tensor(self.original_generator.generaten(unlabeledX, m)).float()
        generatedY = self._generaten(unlabeledX, n)
        generatedBeta = torch.sigmoid((generatedY.unsqueeze(2) - orig_generatedY.unsqueeze(1)) * temperature).mean(dim=2)
        return generatedBeta.detach().numpy()

    def generate_beta_scc(self, unlabeledX:np.ndarray, qrmodel:QRModel, n:int=10):
        orig_generatedY = torch.tensor(qrmodel.predict(unlabeledX)).view(-1, 1).float()
        generatedY = self._generaten(unlabeledX, n)
        generatedBeta = generatedY - orig_generatedY
        return generatedBeta.detach().numpy()

    def tune_marginal(self, labeledX:np.ndarray, labeledY:np.ndarray, unlabeledX:np.ndarray, n:int=10, epochs:int=100, learning_rate:float=0.01, n_grid=50, lbd:float=1., temperature=10., **kwargs):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控最小化指标（如 loss）
            factor=0.9,  # 新 LR = old LR * factor
            patience=50,  # 容忍 50 个 epoch 不改善
        )
        nmin = min(labeledX.shape[0], n_grid)
        q = torch.tensor(np.linspace(1/nmin, 1, nmin)).float()
        m = kwargs.get('m', 200)
        fun_stat = penalty_fun[kwargs.get('penalty', 'MAE')]

        empiricalBeta = torch.tensor(self.original_generator.cdf(labeledX, labeledY, m)).view(-1, 1).float()
        empiricalQ = torch.quantile(empiricalBeta, q, interpolation='higher')

        orig_generatedY = torch.tensor(self.original_generator.generaten(unlabeledX, m)).float()
        if lbd == 0.:
            max_iter, tol_gap, targ_alpha = kwargs.get('max_iter', 1000), kwargs.get('tol_gap', 0.01), kwargs.get('targ_alpha', 0.1)
            q_emp = torch.quantile(empiricalBeta, 1-targ_alpha, interpolation='higher')
            for ep in range(max_iter):
                generatedY = self._generaten(unlabeledX, n)
                generatedBeta = torch.sigmoid((generatedY.unsqueeze(2) - orig_generatedY.unsqueeze(1)) * temperature).mean(dim=2)
                generatedQ = torch.quantile(generatedBeta, q, interpolation='higher')
                L = fun_stat(empiricalQ, generatedQ) + fun_stat(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'),q_emp)
                optimizer.zero_grad()
                if (epochs <= ep) and torch.abs(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher')-q_emp) < tol_gap:
                    return L.detach().item(), self.get_delta_norm().detach().item()
                L.backward()
                optimizer.step()
                scheduler.step(L.detach().item())
            return L.detach().item(), self.get_delta_norm().detach().item()
            # return torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'), q_emp
        else:
            max_iter, tol_gap, targ_alpha = kwargs.get('max_iter', 1000), kwargs.get('tol_gap', 0.01), kwargs.get('targ_alpha', 0.1)
            q_emp = torch.quantile(empiricalBeta, 1-targ_alpha, interpolation='higher')
            for ep in range(epochs):
                generatedY = self._generaten(unlabeledX, n)
                generatedBeta = torch.sigmoid((generatedY.unsqueeze(2) - orig_generatedY.unsqueeze(1)) * temperature).mean(dim=2)
                generatedQ = torch.quantile(generatedBeta, q, interpolation='higher')
                part1 = fun_stat(empiricalQ, generatedQ) + fun_stat(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'),q_emp)
                L = part1 + lbd*self.get_delta_norm()
                optimizer.zero_grad()
                if ep == epochs-1:
                    return part1.detach().item(), self.get_delta_norm().detach().item()
                L.backward()
                optimizer.step()
                scheduler.step(L.detach().item())
            # return torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'), q_emp

    def tune_marginal_scc(self, labeledX: np.ndarray, labeledY: np.ndarray, unlabeledX: np.ndarray, qrmodel:QRModel, n: int = 10, epochs: int = 100, learning_rate: float = 0.01, n_grid=50, lbd: float = 1., **kwargs):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控最小化指标（如 loss）
            factor=0.9,  # 新 LR = old LR * factor
            patience=50,  # 容忍 50 个 epoch 不改善
        )
        nmin = min(labeledX.shape[0], n_grid)
        q = torch.tensor(np.linspace(1 / nmin, 1, nmin)).float()
        fun_stat = penalty_fun[kwargs.get('penalty', 'MAE')]

        empiricalY = torch.tensor(labeledY).view(-1, 1).float()
        empiricalBeta = empiricalY - torch.tensor(qrmodel.predict(labeledX)).view(-1, 1).float()
        empiricalQ = torch.quantile(empiricalBeta, q, interpolation='higher')

        orig_generatedY = torch.tensor(qrmodel.predict(unlabeledX)).view(-1, 1).float()
        if lbd == 0.:
            max_iter, tol_gap, targ_alpha = kwargs.get('max_iter', 1000), kwargs.get('tol_gap', 0.01), kwargs.get('targ_alpha', 0.1)
            q_emp = torch.quantile(empiricalBeta, 1-targ_alpha, interpolation='higher')
            for ep in range(max_iter):
                generatedY = self._generaten(unlabeledX, n)
                generatedBeta = generatedY - orig_generatedY
                generatedQ = torch.quantile(generatedBeta, q)
                L = fun_stat(empiricalQ, generatedQ) + fun_stat(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'),q_emp)
                optimizer.zero_grad()
                if (epochs <= ep) and torch.abs(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher')-q_emp) < tol_gap:
                    return L.detach().item(), self.get_delta_norm().detach().item()
                L.backward()
                optimizer.step()
                scheduler.step(L.detach().item())
            return L.detach().item(), self.get_delta_norm().detach().item()
            # return torch.quantile(generatedBeta, 1 - targ_alpha, interpolation='higher'), q_emp
        else:
            max_iter, tol_gap, targ_alpha = kwargs.get('max_iter', 1000), kwargs.get('tol_gap', 0.01), kwargs.get('targ_alpha', 0.1)
            q_emp = torch.quantile(empiricalBeta, 1 - targ_alpha, interpolation='higher')
            for ep in range(epochs):
                generatedY = self._generaten(unlabeledX, n)
                generatedBeta = generatedY - orig_generatedY
                generatedQ = torch.quantile(generatedBeta, q)
                part1 = fun_stat(empiricalQ, generatedQ) + fun_stat(torch.quantile(generatedBeta, 1-targ_alpha, interpolation='higher'),q_emp)
                L = part1 + lbd*self.get_delta_norm()
                optimizer.zero_grad()
                if ep == epochs - 1:
                    return part1.detach().item(), self.get_delta_norm().detach().item()
                L.backward()
                optimizer.step()
                scheduler.step(L.detach().item())

if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)
    X = np.random.uniform(-4, 4, 200)
    Y = X + np.cos(X) * np.random.normal(0, 1, 200)
    generator = Generator(1, [20, 100, 100, 20], randNum=3)
    generator.trainEng(X.reshape((-1, 1)), Y.reshape((-1, 1)), m=10, batch_size=32, epochs=100, log=log, mute=False)

    tuner = Tuner(generator, [1])
    X_ = np.repeat(X, 5, axis=0)
    yhat = tuner.generaten(X.reshape((-1, 1)), 5)

    setseed(5)
    tuner = Tuner(generator, [1])
    labeledX = X[:50]
    labeledY = labeledX + 1.2 * np.cos(labeledX) * np.random.normal(0, 1, 50)
    unlabeledX = np.random.uniform(-4, 4, 1000)

    print(np.quantile(generator.cdf(labeledX, labeledY, 500), 0.9, interpolation='higher'))
    generatedY = tuner.generaten(unlabeledX, 5)
    print(np.quantile(generator.cdf(unlabeledX, generatedY, 500), 0.9, interpolation='higher'))
    print(tuner.tune_marginal(labeledX, labeledY, unlabeledX, n=5, learning_rate=0.01, lbd=0., temperature=10, epochs=100, m=500, max_iter=5000, tol_gap=0.005, penalty='MAE'))
    beta_ = tuner.generate_beta(unlabeledX, 500, 5, 10.)
    print(np.quantile(beta_, 0.9, interpolation='higher'), tuner.get_delta_norm().detach())
