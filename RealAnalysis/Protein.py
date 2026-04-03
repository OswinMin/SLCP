import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tools import *
from procedure import *
import warnings
import datetime
from copy import deepcopy
import ast
import itertools
import pickle
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'RealAnalysis':
        os.chdir(os.path.join(os.getcwd(), 'RealAnalysis'))
    protdata = pd.read_csv("../Dataset/proteinStructure.csv")
    X_raw = protdata.drop('RMSD', axis=1)
    y_raw = np.log(protdata['RMSD'].values+1)
    d = X_raw.shape[1]

    # each agent has 2*n samples
    n, m, N, tar_ind, n_grids, lbds, temperatures = 60, 1000, 2000, 4, [50], [0.005, 0.03], [10.]
    isLog = True
    seed, epoches, repeats, testN, alpha = 1, 200, 30, 3000, 0.1
    hidden_dim, noise_dim = [20, 50, 30, 20], d  # Engression parameters
    X_test_group = [10, 15, 20, 25, 30]
    Y_int = 10

    SimRpath = f"../SimResult/Real_Protein"
    SimName = f"P_{n}_{m}_{N}_{tar_ind}"
    Lpath = f"../Log/Real_Protein/{n}_{m}_{N}_{tar_ind}.txt"
    SumLpath = f"../Log/Real_Protein/Sum_{n}_{m}_{N}_{tar_ind}.txt"

    setseed(seed)
    X_transformed, skewed_cols = auto_skew_transform(X_raw.copy(), log, '', False, skew_threshold=1.0)
    scaler = StandardScaler()
    X_trans = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X_raw.columns).values
    nLoc = [i / Y_int for i in range(Y_int + 1)]
    y_loc = np.quantile(y_raw, nLoc)
    y_scale = [(y_loc[i + 1] - y_loc[i - 1]) / 2 for i in range(1, Y_int)]
    y_scale = [y_scale[0]] + y_scale + [y_scale[-1]]
    Xt, yt, X, y = splitData(X_trans, y_raw, selectN(y_raw, y_loc[tar_ind], y_scale[tar_ind], testN+m+n))
    agent_target = Agent(d, testN+m+n, Xt, yt)
    agent_aux = Agent(d, X.shape[0], X, y)
    addDict = {"Y_group":Y_int, "tar_ind":tar_ind, "X_test_group":X_test_group}

    procedure_reg(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, hidden_dim, noise_dim, epoches, alpha, agent_target, agent_aux, seed=seed, Lpath=Lpath, SumLpath=SumLpath, SimRpath=SimRpath, SimName=SimName, isLog=isLog, addDict=addDict)
