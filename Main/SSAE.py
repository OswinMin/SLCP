import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Union

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        :param input_dim: 输入维度 (d)
        :param hidden_dims: list, 中间隐藏层维度列表，例如 [128, 64, 32]
        """
        super().__init__()
        self.input_dim = input_dim
        encoder_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            curr_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        layer_dims = [hidden_dims[-1]] + hidden_dims[:-1][::-1] + [input_dim]
        for i in range(len(layer_dims) - 1):
            decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_loop(self, X, epochs=100, lr=1e-3, batch_size=32, verbose=True, **kwargs):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.input_dim)
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                x_batch = batch[0]

                optimizer.zero_grad()
                x_recon = self.forward(x_batch)
                loss = criterion(x_recon, x_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            report_freq = kwargs.get('report_num', 5)
            if verbose and (epoch + 1) % max(1, epochs // report_freq) == 0:
                print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        return self

    def residual_predict(self, X):
        """
        预测残差, X: n*d    np.ndarray or torch.Tensor
        return: n   np.ndarray
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.input_dim)
        self.eval()
        with torch.no_grad():
            X_recon = self.forward(X)
            residuals = ((X - X_recon) ** 2).sum(dim=1).sqrt()
        return residuals.numpy()

    def extend_features(self, X):
        """
        预测残差, X: n*d    np.ndarray or torch.Tensor
        return: n*(d+1)     np.ndarray
        """
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        X = X.reshape((-1, self.input_dim))
        residuals = self.residual_predict(X).reshape((-1, 1))
        extended_features = np.concatenate([X, residuals], axis=1)
        return extended_features

    def extract_features(self, X):
        """
        提取编码器的特征, X: n*d    np.ndarray or torch.Tensor
        return: n*hidden_dims[-1]     np.ndarray
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.input_dim)
        self.eval()
        with torch.no_grad():
            features = self.encoder(X)
        return features.numpy()

    def save(self, path):
        """保存模型参数到指定路径"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """从指定路径加载模型参数"""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
        return self

class semiQuantile:
    def __init__(self, inputDim, **kwargs):
        from engGenerator import Generator
        self.inputDim = inputDim
        self.model = Generator(inputDim=inputDim, **kwargs)

    def fit(self, X:np.ndarray, Y:np.ndarray, unX:np.ndarray, **kwargs):
        self.model.trainEng(X.reshape((-1, self.inputDim)), Y.reshape((-1, 1)), **kwargs)
        self.X_all = np.concatenate([X.reshape((-1, self.inputDim)), unX.reshape((-1, self.inputDim))], axis=0)
        self.n = X.shape[0]
        self.N = self.X_all.shape[0]
        self.Y = Y.reshape((-1, 1))
        self.m = kwargs.get('m', 100) # 生成样本数量
        return self

    def fit_outer(self, X:np.ndarray, Y:np.ndarray, unX:np.ndarray, outerX:np.ndarray, outerY:np.ndarray, **kwargs):
        self.model.trainEng(outerX.reshape((-1, self.inputDim)), outerY.reshape((-1, 1)), **kwargs)
        self.X_all = np.concatenate([X.reshape((-1, self.inputDim)), unX.reshape((-1, self.inputDim))], axis=0)
        self.n = X.shape[0]
        self.N = self.X_all.shape[0]
        self.Y = Y.reshape((-1, 1))
        self.m = kwargs.get('m', 100) # 生成样本数量
        return self

    def cdf(self, y):
        if not isinstance(y, np.ndarray):
            y = np.ones((self.N, 1)) * y
        else:
            y = np.ones((self.N, 1)) * y.reshape((1, -1))
        gen_cdf = self.model.cdf(self.X_all, y, self.m).reshape(-1)
        return gen_cdf.mean() + (self.Y <= y[:self.n]).mean() - gen_cdf[:self.n].mean()

    def quantile(self, q, **kwargs):
        low, high = self.Y.min(), self.Y.max()
        for _ in range(kwargs.get('max_iter', 100)):
            mid = (low + high) / 2
            if self.cdf(mid) < q:
                low = mid
            else:
                high = mid
        return high

    def quantile_raw(self, q):
        return np.quantile(self.Y, q, method='higher')

from GLCP import *

class dissemiGLCP(GLCP):
    def __init__(self, targetAgent: Agent, generator: Generator, predictor: Predictor, unX = None, n: int = 1000, alpha: float = 0.1, outer=False, outerX=None, outerbeta=None, iftune=False, **kwargs):
        super().__init__(targetAgent, generator, predictor, n, alpha)
        self.sq = semiQuantile(inputDim=targetAgent.d, hiddenDim=kwargs.get('hiddenDim', [20, 20]), randNum=targetAgent.d)
        if outer:
            self.sq.fit_outer(targetAgent.getX(), self.beta, unX, outerX=outerX, outerY=outerbeta, batch_size=kwargs.get('batch_size', 32), epochs=kwargs.get('epochs', 300), learning_rate=kwargs.get('learning_rate', 0.01), m=kwargs.get('m', 100), mute=kwargs.get('mute', True))
        else:
            self.sq.fit(targetAgent.getX(), self.beta, unX, batch_size=kwargs.get('batch_size', 32), epochs=kwargs.get('epochs', 300), learning_rate=kwargs.get('learning_rate', 0.01), m=kwargs.get('m', 100), mute=kwargs.get('mute', True))
        if iftune:
            self.q = self.sq.quantile((1-alpha)*(targetAgent.n+1)/targetAgent.n)
        else:
            self.q = self.sq.quantile(1-alpha)

class dissemiSCC(SCC):
    def __init__(self, targetAgent: Agent, predictor: QRModel, basePredictor: Predictor, unX = None, alpha: float = 0.1, outer=False, outerX=None, outerbeta=None, iftune=False, **kwargs):
        super().__init__(targetAgent, predictor, basePredictor, alpha)
        self.sq = semiQuantile(inputDim=targetAgent.d, hiddenDim=kwargs.get('hiddenDim', [20, 20]), randNum=targetAgent.d)
        if outer:
            self.sq.fit_outer(targetAgent.getX(), self.beta, unX, outerX=outerX, outerY=outerbeta, batch_size=kwargs.get('batch_size', 32), epochs=kwargs.get('epochs', 300), learning_rate=kwargs.get('learning_rate', 0.01), m=kwargs.get('m', 100), mute=kwargs.get('mute', True))
        else:
            self.sq.fit(targetAgent.getX(), self.beta, unX, batch_size=kwargs.get('batch_size', 32), epochs=kwargs.get('epochs', 300), learning_rate=kwargs.get('learning_rate', 0.01), m=kwargs.get('m', 100), mute=kwargs.get('mute', True))
        if iftune:
            self.qhat = self.sq.quantile((1-alpha)*(targetAgent.n+1)/targetAgent.n)
        else:
            self.qhat = self.sq.quantile(1-alpha)

if __name__ == "__main__":
    n, d = 64, 10
    X = torch.randn(n, d)
    unX = torch.randn(4*n, d)
    Y = X.sum(dim=1) + torch.randn(n) * 0.1

    model = AutoEncoder(input_dim=d, hidden_dims=[32, 5])

    model.train_loop(X, epochs=1000)
    sX = model.residual_predict(X)

    X_extended = model.extend_features(X)
    print("Extended features shape:", X_extended.shape)  # 应该是 (n,d+1)

    sq = semiQuantile(inputDim=d, hiddenDim=[20,20], randNum=d)
    sq.fit(X.numpy(), Y.numpy(), unX.numpy(), batch_size=32, epochs=300, learning_rate=0.01, m=100)
    print("Quantile at 0.5:", sq.quantile(0.5), sq.quantile_raw(0.5))