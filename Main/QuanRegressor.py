import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Union
from tools import *

class QRModel:
    def __init__(self, methods:str='Engressor', q:float=.9, **kwargs):
        self.methods = methods
        self.q = q
        self.init_kwargs = kwargs
        if methods == 'Engressor':
            from engGenerator import Generator
            self.model = Generator(**kwargs)
        elif methods == 'Linear':
            from sklearn.linear_model import QuantileRegressor
            self.model = QuantileRegressor(quantile=q, alpha=kwargs.get('alpha', 0.0), solver=kwargs.get('solver', 'highs'))
        elif methods == 'RF':
            from sklearn_quantile import RandomForestQuantileRegressor
            self.model = RandomForestQuantileRegressor(n_estimators=kwargs.get('n_estimators', 100), max_depth=kwargs.get('max_depth', None), random_state=kwargs.get('random_state', 42), min_samples_split=kwargs.get('min_samples_split', 2), q=q)
        elif self.methods == 'LGB':
            self.params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'num_leaves': kwargs.get('num_leaves', 31),
                'learning_rate': kwargs.get('learning_rate', 0.05),
                'feature_fraction': kwargs.get('feature_fraction', 0.8),
                'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
                'bagging_freq': kwargs.get('bagging_freq', 5),
                'verbose': kwargs.get('verbose', -1)
            }
        elif self.methods == 'CDF':
            self.CDFModel = kwargs.get('CDFModel', None)    # An already trained model
            self.d = kwargs.get('d', None)
        else:
            raise NotImplementedError(f"Method {methods} not implemented yet.")

    def fit(self, X, y, **kwargs):
        self.d = X.shape[1]
        if self.methods == 'Engressor':
            self.model.trainEng(X, y.reshape((-1, 1)), **kwargs)
            self.init_kwargs['m'] = kwargs.get('m', 100)
        elif self.methods == 'Linear':
            self.model.fit(X, y.reshape(-1))
        elif self.methods == 'RF':
            self.model.fit(X, y.reshape(-1))
        elif self.methods == 'LGB':
            import lightgbm as lgb
            train_data = lgb.Dataset(X, label=y.reshape(-1))
            self.model = lgb.train(self.params, train_data, num_boost_round=kwargs.get('num_boost_round', 100), callbacks=[lgb.early_stopping(10)] if kwargs.get('early_stopping', True) else None)

    def predict(self, X:np.ndarray, **kwargs):
        """
        :param X: n * d
        :param kwargs:
        :return: (n, 1) quantile prediction
        """
        X = X.reshape((-1, self.d))
        if self.methods == 'Engressor':
            return self.model.quantile(X, q=self.q, m=self.init_kwargs.get('m', 100)).reshape((-1, 1))
        elif self.methods == 'Linear':
            return self.model.predict(X).reshape((-1, 1))
        elif self.methods == 'RF':
            return self.model.predict(X).reshape((-1, 1))
        elif self.methods == 'LGB':
            return self.model.predict(X).reshape((-1, 1))
        elif self.methods == 'CDF':
            return self.CDFModel.quantile(X, q=self.q, **kwargs).reshape((-1, 1))

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 1000
    X = torch.randn(n_samples, 1) * 2
    y = (X**2).squeeze() + 0.3 * torch.randn(n_samples)
    eng = QRModel('Engressor', 0.5, inputDim=1, hiddenDim=[32, 32], restrict=False, randNum=1)
    eng.fit(X, y, m=100, batch_size=32, epochs=100, learning_rate=1e-2)

    X_test = torch.linspace(-5, 5, 30).unsqueeze(1)
    y_pred_median = eng.predict(X_test.numpy()).reshape(-1)
    y_median = (X_test**2).squeeze().numpy()
    print(np.abs(y_median - y_pred_median).mean())
