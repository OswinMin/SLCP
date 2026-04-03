import copy
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from CNNnet import *
from tools import *
from procedure import *
import warnings
import datetime
from copy import deepcopy
import ast
import itertools
import pickle
warnings.filterwarnings('ignore')

def proc_feat_resp(X:np.ndarray):
    s = list(X.shape)
    s.append(1)
    X = X.reshape(s)
    return X

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'RealAnalysis':
        os.chdir(os.path.join(os.getcwd(), 'RealAnalysis'))

    n, m, N, w0, n_grids, lbds, temperatures, isLog = 60, 1000, 2000, 0.07, [50], [0.3], [10.], True
    seed, testN = 1, 500
    setseed(seed)
    data = np.load('../Dataset/tissuemnist.npz')
    images = proc_feat_resp(data['val_images'])
    labels = data['val_labels']
    typeCount = np.unique(labels, return_counts=True)
    typeNum = len(typeCount[0])
    tar_w = [1/9+w0] + [2/15-3*w0/5]*5 + [1/9+w0]*2
    tar_N = [min(int((n+m+testN)*tar_w[i]), typeCount[1][i] // 3) for i in range(len(tar_w))]
    tar_N[0] = n+m+testN-np.sum(tar_N)+tar_N[0]
    # [int((n+m+testN)*typeCount[1][i]/np.sum(typeCount[1])) for i in range(typeNum)]
    aux_N = typeCount[1] - np.array(tar_N)
    agent_target, agent_aux = selectTarget(images, labels, tar_N, aux_N)

    epoches, repeats, d, alpha = 200, 30, 10, 0.1
    hidden_dim, noise_dim = [20, 50, 30, 20], d
    in_channels = 1
    trainerMap = MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        num_epochs=100,
        num_classes=typeNum,
        in_channels=1
    )
    trainerMap.load("Para/TissueMNIST.pth")

    SimRpath = f"../SimResult/Real_Tissue"
    SimName = f"P_{n}_{m}_{N}_{w0}"
    Lpath = f"../Log/Real_Tissue/{n}_{m}_{N}_{w0}.txt"
    SumLpath = f"../Log/Real_Tissue/Sum_{n}_{m}_{N}_{w0}.txt"
    addDict = {'target data class ratio':tar_w,
               'target data class number':np.array(tar_N),
               'souce data class number':np.array(aux_N),}
    model_path = "Para/TissueMNIST.pth"

    procedure_cls(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, typeNum, hidden_dim, noise_dim, epoches, alpha, agent_target, agent_aux, in_channels=in_channels, seed=seed, Lpath=Lpath, SumLpath=SumLpath, SimRpath=SimRpath, SimName=SimName, isLog=isLog, addDict=addDict, model_path=model_path, predmethod='rf')