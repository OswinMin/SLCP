import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as rfr
from tools import *
from Agents import *


class ThreeLayer(nn.Module):
    def __init__(self, inputDim:int=1, hiddenDim:list[int]=(1,1), **kwargs):
        super(ThreeLayer, self).__init__()
        self.l1 = nn.Linear(inputDim, hiddenDim[0])
        self.l2 = nn.Linear(hiddenDim[0], hiddenDim[1])
        self.l3 = nn.Linear(hiddenDim[1], 1)
        self.R = nn.ReLU()
    def forward(self, x:torch.Tensor):
        """
        :param x: n*inputDim
        :return: n*1
        """
        x = self.R(self.l1(x))
        x = self.R(self.l2(x))
        return self.l3(x)
    def train_(self, X:np.ndarray, Y:np.ndarray, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, suppress:bool=False, **kwargs):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch % (epochs // 10) == max((epochs // 10) - 1, 1)) and (not suppress):
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    def trainFromAgent(self, agent:Agent, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, suppress:bool=False, **kwargs):
        self.train_(agent.X, agent.Y, batch_size, epochs, learning_rate, suppress)
    def predict(self, x:np.ndarray):
        return self.forward(torch.tensor(x).float()).detach().numpy()


class LReg:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
    def train_(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        self.model.fit(X, Y)
    def predict(self, x:np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            return self.model.predict(x)[0]
        else:
            return self.model.predict(x).reshape((-1, 1))
    def trainFromAgent(self, agent: Agent, **kwargs):
        self.train_(agent.X, agent.Y)


class SimpleTree:
    def __init__(self, n_estimators:int=100, max_depth=3, min_samples_split=2, **kwargs):
        self.model = rfr(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    def train_(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        self.model.fit(X, Y.reshape(-1))
    def predict(self, x: np.ndarray):
        return self.model.predict(x).reshape((-1, 1))
    def trainFromAgent(self, agent: Agent, **kwargs):
        self.train_(agent.X, agent.Y)

class ZeroPredictor:
    def __init__(self, **kwargs):
        pass
    def train_(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        pass
    def predict(self, x: np.ndarray):
        if len(x.shape) == 1:
            return np.array([[0]])
        else:
            return np.zeros(x.shape[0]).reshape((-1, 1))
    def trainFromAgent(self, agent: Agent, **kwargs):
        self.train_(agent.X, agent.Y)

class Predictor:
    def __init__(self, method="nn", **kwargs):
        """
        :param method: nn (network), lr (linear regression), rf (random forest)
        :param kwargs required:
        method == nn:
            inputDim:int=1, (dimension d)
            hiddenDim:list[int]=(1,1), hidden layer length
        method == rf:
            n_estimators:int=100,
            max_depth=3,
            min_samples_split=2,
        """
        self.method = method
        if self.method == 'nn':
            self.model = ThreeLayer(**kwargs)
        elif self.method == 'lr':
            self.model = LReg(**kwargs)
        elif self.method == 'rf':
            self.model = SimpleTree(**kwargs)
        elif self.method == 'zero':
            self.model = ZeroPredictor()
        elif self.method == 'partial':
            self.model = kwargs.get('model', None)
            self.firstk = kwargs.get('firstk', None)
        else:
            self.model = None

    def predict(self, x:np.ndarray):
        """
        :param x: n*inputDim
        :return: n*1 np.ndarray
        """
        if self.method == 'partial':
            return self.model.predict(x[:, :self.firstk])
        else:
            return self.model.predict(x)

    def train(self, X:np.ndarray, Y:np.ndarray, **kwargs):
        """
        Use X, Y train a simple predictor
        :param X: n*inputDim
        :param Y: n*1
        :param kwargs required:
        method == nn:
            batch_size:int=32,
            epochs:int=100,
            learning_rate:float=0.01,
            suppress:False (whether to print loss per epochs/10 or not)
        """
        self.model.train_(X, Y, **kwargs)

    def trainFromAgent(self, agent:Agent, **kwargs):
        self.model.trainFromAgent(agent, **kwargs)