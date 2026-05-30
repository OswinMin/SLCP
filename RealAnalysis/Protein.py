import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
from tools import *
from sklearn.preprocessing import StandardScaler
from procedure import *
import warnings
import ast
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'RealAnalysis':
        os.chdir(os.path.join(os.getcwd(), 'RealAnalysis'))
    protdata = pd.read_csv("../Dataset/proteinStructure.csv")
    X_raw = protdata.drop('RMSD', axis=1)
    y_raw = np.log(protdata['RMSD'].values+1)
    d = X_raw.shape[1]

    # each agent has 2*n samples
    n, m, N, tar_ind, n_grids, lbds, temperatures = 60, 1000, 2000, 4, [50], [0., 0.002, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05], [10.]
    if len(sys.argv) > 1:
        n, m, N, tar_ind, n_grids, lbds, temperatures = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), ast.literal_eval(sys.argv[5]), ast.literal_eval(sys.argv[6]), ast.literal_eval(sys.argv[7])
    isLog = True
    seed, epoches, repeats, testN, alpha = 1, 100, 50, 500, 0.1
    hidden_dim, noise_dim = [50, 100, 100, 50], d  # Engression parameters
    Y_int = 10

    setseed(seed)
    X_transformed, skewed_cols = auto_skew_transform(X_raw.copy(), log, '', False, skew_threshold=1.0)
    scaler = StandardScaler()
    X_trans = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X_raw.columns).values
    # raise KeyboardInterrupt
    nLoc = [0.005 + 0.99 * i / Y_int for i in range(Y_int + 1)]
    y_loc = np.quantile(y_raw, nLoc)
    y_scale = [(y_loc[i + 1] - y_loc[i - 1]) / 2 for i in range(1, Y_int)]
    y_scale = [y_scale[0]] + y_scale + [y_scale[-1]]
    Xt, yt, X, y = splitData(X_trans, y_raw, selectN(y_raw, y_loc[tar_ind], y_scale[tar_ind], testN+m+n))
    agent_target = Agent(d, testN+m+n, Xt, yt)
    agent_aux = Agent(d, X.shape[0], X, y)
    addDict = {"Y_group":Y_int, "tar_ind":tar_ind}
    N = min(N, agent_aux.n)

    SimRpath = f"../SimResult/Real_Protein"
    SimName = f"P_{n}_{m}_{N}_{tar_ind}"
    Lpath = f"../Log/Real_Protein/{n}_{m}_{N}_{tar_ind}.txt"
    SumLpath = f"../Log/Real_Protein/Sum_{n}_{m}_{N}_{tar_ind}.txt"

    procedure_reg(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, hidden_dim, noise_dim, epoches, alpha, agent_target, agent_aux, tol_gap=2e-2, seed=seed, Lpath=Lpath, SumLpath=SumLpath, SimRpath=SimRpath, SimName=SimName, isLog=isLog, addDict=addDict, comb_aux_tr=True, predmethod='rf')
