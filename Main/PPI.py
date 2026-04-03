import numpy as np
from Predictor import *

class PPI_quantile:
    def __init__(self, q, gap=1e-12):
        self.q = q
        self.gap = gap

    def fit(self, X, y, unX, f):
        Yhat = f.predict(X).reshape(-1)
        unYhat = f.predict(unX).reshape(-1)
        Y = y.reshape(-1)
        n, N = len(Y), len(unYhat)

        def F(u):
            return (np.sum(unYhat <= u) / N + (np.sum(Y <= u) - np.sum(Yhat <= u)) / n)
        all_vals = np.concatenate([Y, Yhat, unYhat])
        lo, hi = all_vals.min() - 1, all_vals.max() + 1
        while hi - lo > self.gap:
            mid = (lo + hi) / 2
            if F(mid) >= self.q:
                hi = mid
            else:
                lo = mid
        return hi