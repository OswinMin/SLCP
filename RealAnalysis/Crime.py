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
    """
    Split agent using population
    Target: 2 choices (highest or lowest)
    Rest data serve as auxiliary agent
    use all covariates by kmeans to split the support of X to calculate conditional coverage
    """
    if os.path.split(os.getcwd())[1] != 'RealAnalysis':
        os.chdir(os.path.join(os.getcwd(), 'RealAnalysis'))
    # noinspection PyTypeChecker
    crimedata = pd.read_csv("../Dataset/communitiesCrime/crimedata.csv", sep='\s*,\s*', encoding='latin-1', engine='python', na_values=["?"])
    crimedata = crimedata[~crimedata['ViolentCrimesPerPop'].isna()]
    cols = ['HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'PctUnemployed', 'PctEmploy', 'murdPerPop', 'rapesPerPop', 'robbbPerPop', 'assaultPerPop', 'burglPerPop', 'larcPerPop', 'autoTheftPerPop', 'arsonsPerPop', 'nonViolPerPop']
    crimedata.fillna(crimedata[cols].median(), inplace=True)
    crimedata = crimedata.dropna(axis=1, how='any')
    crimedata = crimedata.rename(columns={'Êcommunityname': 'communityName'})
    cols = ['population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct65up', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'HispPerCap', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PctForeignBorn', 'PctBornSameState', 'PopDens', 'LemasPctOfficDrugUn']
    X_raw, y = crimedata[cols], np.log(crimedata['ViolentCrimesPerPop']+1).values
    importance = [np.abs(np.corrcoef(X_raw.iloc[:,i], y)[0,1]) for i in range(1,X_raw.shape[1])]
    scaler = StandardScaler()
    X_raw = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns).values

    n, m, high, features, n_grids, lbds, temperatures, isLog = 60, 500, 0, 15, [50], [0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02], [10.], True
    if len(sys.argv) > 1:
        n, m, high, features, n_grids, lbds, temperatures = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), ast.literal_eval(sys.argv[5]), ast.literal_eval(sys.argv[6]), ast.literal_eval(sys.argv[7])
        isLog = True

    seed, testN = 0, 100
    N = X_raw.shape[0] - n - m - testN
    setseed(seed)
    importance_indice = np.argsort(importance)
    X = np.concatenate([X_raw[:,[0]], (X_raw[:, 1:].T[importance_indice<features]).T], axis=1)
    if high == 1:
        target_mask = X[:, 0] >= np.quantile(X[:, 0], 1-(n+m+testN)/X.shape[0])
    else:
        target_mask = X[:, 0] <= np.quantile(X[:, 0], (n+m+testN)/X.shape[0])
    agent_target, agent_aux = Agent(features, np.sum(target_mask), X[target_mask][:, 1:], y[target_mask]), Agent(features, np.sum(~target_mask), X[~target_mask][:, 1:], y[~target_mask])

    epoches, repeats, alpha = 100, 50, 0.1
    hidden_dim, noise_dim, d = [50, 100, 100, 50], features, features  # Engression parameters

    SimRpath = f"../SimResult/Real_Crime"
    SimName = f"P_{n}_{m}_{N}_{high}_{features}"
    Lpath = f"../Log/Real_Crime/{n}_{m}_{N}_{high}_{features}.txt"
    SumLpath = f"../Log/Real_Crime/Sum_{n}_{m}_{N}_{high}_{features}.txt"
    addDict = {'high':high, 'features':features}

    procedure_reg(n_grids, lbds, temperatures, repeats, testN, m, n, N, d, hidden_dim, noise_dim, epoches, alpha, agent_target, agent_aux, tol_gap=2e-2, seed=seed, Lpath=Lpath, SumLpath=SumLpath, SimRpath=SimRpath, SimName=SimName, isLog=isLog, addDict=addDict, predmethod='rf', comb_aux_tr=True)
