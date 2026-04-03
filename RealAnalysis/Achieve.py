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

def defaultSolveScore(x, s, predictor):
    """
    :param x: [d]
    :param s:
    :param predictor:
    :return:
    """
    if len(x.shape) == 1:
        yhat = predictor.predict(x).item()
        return np.clip(np.array([yhat - s, yhat + s]), 0., 136.), False
    else:
        yhat = predictor.predict(x).reshape(-1)
        cs = np.zeros((yhat.shape[0], 2))
        cs[:, 0] = yhat - s.reshape(-1)
        cs[:, 1] = yhat + s.reshape(-1)
        return np.clip(cs, 0., 136.), np.zeros(cs.shape[0])

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'RealAnalysis':
        os.chdir(os.path.join(os.getcwd(), 'RealAnalysis'))
    n, m, features, targ_g, targ_ind, n_grids, lbds, temperatures, isLog = 60, 500, 15, 2, 2, [50], [0.002, 0.1], [10.], True
    # noinspection PyTypeChecker
    achievedata = pd.read_spss("../Dataset/achievementRatio/STAR_Students.sav")
    achievedata = achievedata[~achievedata['hsacttot'].isna()]
    for col in achievedata.columns:
        if (col[:4] == 'flag') or (col[:4] == 'FLAG') or (col[-2:] == 'id') or (col[-4:] == 'tgen'):
            achievedata.drop(col, axis=1, inplace=True)
    achievedata = achievedata[achievedata.columns[achievedata.isna().mean(0)<.8]]
    y = achievedata['hsacttot'].values
    basic_col = ['gender', 'race', 'birthyear']
    clst_col = ['g1classtype', 'g2classtype', 'g3classtype', 'cmpstype', 'cmpsdura', 'yearstar', 'yearssmall']
    clssize_col = ['g1classsize', 'g2classsize', 'g3classsize']
    freel_col = ['g1freelunch', 'g2freelunch', 'g3freelunch']
    ss_col = []
    for i in [2,3,4,5,6,7,8]:
        if f'g{i}treadss' in achievedata.columns:
            ss_col.append(f'g{i}treadss')
        if f'g{i}tmathss' in achievedata.columns:
            ss_col.append(f'g{i}tmathss')
        if f'g{i}tlangss' in achievedata.columns:
            ss_col.append(f'g{i}tlangss')
    obj_col = ['gender', 'race', 'g1classtype', 'g2classtype', 'g3classtype', 'cmpstype', 'g1freelunch', 'g2freelunch', 'g3freelunch']
    seq_col = ['birthyear', 'cmpsdura', 'yearssmall', 'g1classsize', 'g2classsize', 'g3classsize'] + ss_col
    cols = obj_col + seq_col

    group_col = [f'g{targ_g}surban']
    target_mask = np.ones((achievedata.shape[0],), dtype=bool)
    if targ_ind == 0:
        target_mask = target_mask & ((achievedata[group_col[0]] == 'INNER CITY')|(achievedata[group_col[0]] == 'URBAN'))
    elif targ_ind == 1:
        target_mask = target_mask & (achievedata[group_col[0]] == 'RURAL')
    else:
        target_mask = target_mask & (achievedata[group_col[0]] == 'SUBURBAN')

    achievedata = achievedata[cols]
    for col in obj_col:
        cnt = achievedata[col].cat.codes.value_counts()
        mapping = {}
        for i in range(cnt.shape[0]):
            if cnt.iloc[i] > achievedata.shape[0] * .1:
                mapping[cnt.index[i]] = i
            else:
                mapping[cnt.index[i]] = -1
        achievedata[col] = achievedata[col].cat.codes.map(mapping).fillna(-1).astype(float)
    for col in seq_col:
        mean_value = achievedata[col].mean()
        achievedata[col].fillna(mean_value, inplace=True)
        achievedata[col] = (achievedata[col] - achievedata[col].mean()) / achievedata[col].std()
    X_raw = achievedata.values.copy()
    importance = [np.abs(np.corrcoef(X_raw[:,i], y)[0,1]) for i in range(X_raw.shape[1])]
    importance_indice = np.argsort(importance)[::-1]
    X = (X_raw[:, importance_indice[:features]])
    # raise KeyError

    agent_target, agent_aux = Agent(features, np.sum(target_mask), X[target_mask], y[target_mask]), Agent(features, np.sum(~target_mask), X[~target_mask], y[~target_mask])
    seed, testN = 0, 200
    d, N = features, agent_aux.n
    epoches, repeats, alpha = 300, 30, 0.1
    hidden_dim, noise_dim = [20, 50, 30, 20], d  # Engression parameters

    SimRpath = f"../SimResult/Real_Star"
    SimName = f"P_{n}_{m}_{N}_{features}_{targ_g}_{targ_ind}"
    Lpath = f"../Log/Real_Star/{n}_{m}_{N}_{features}_{targ_g}_{targ_ind}.txt"
    SumLpath = f"../Log/Real_Star/Sum_{n}_{m}_{N}_{features}_{targ_g}_{targ_ind}.txt"
    addDict = {'features': features, 'targ_g': targ_g, 'targ_ind': targ_ind}

    procedure_reg(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, hidden_dim, noise_dim, epoches, alpha, agent_target, agent_aux, defaultSolveScore=defaultSolveScore, predmethod='rf', seed=seed, Lpath=Lpath, SumLpath=SumLpath, SimRpath=SimRpath, SimName=SimName, isLog=isLog, addDict=addDict, comb_aux_tr=True)
