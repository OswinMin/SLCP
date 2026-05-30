import pickle
import numpy as np

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        resDict = pickle.load(f)
    return resDict

def sum_compare_result(resDict, name_, tol=0.02, n=30, cls=False):
    res = np.zeros((7, 4))
    key_list = [f'{name_}', f'{name_} SDCP', f'{name_} PPI', f'{name_} SLCP', f'{name_} SLCP-sel', f'{name_} ORCP', f'{name_} NOAL']
    for idx in [0, 1, 2, 4, 5, 6]:
        values = resDict[key_list[idx]]
        res[idx, :] = np.array([values[0], values[2], values[3], values[4][0]]) if not cls else np.array([values[0], values[2], values[3], values[4]])
    tol_std = [res[0, 2]]
    for idx in [1, 2]:
        if 0.9-tol <= res[idx, 0] <= 0.9+1/n:
            tol_std.append(res[idx, 2])
    tol_min_std = np.min(tol_std)
    for i in [3]:
        mar, _, size, std, mis = resDict[key_list[i]]
        mask = (std < tol_min_std) & (std >= res[-1, 2]) & (mar >= 0.9-tol) & (mar <= 0.9+1/n)
        if np.sum(mask) == 0:
            mask = (mar >= 0.9-tol) & (mar <= 0.9+1/n)
        if np.sum(mask) == 0:
            idx = np.argmin(np.abs(np.array(mar) - 0.9))
            res[i, :] = (mar[idx], size[idx], std[idx], mis[idx, 0]) if not cls else (mar[idx], size[idx], std[idx], mis[idx])
        else:
            mar, size, std, mis = np.array(mar)[mask], np.array(size)[mask], np.array(std)[mask], np.array(mis)[mask]
            idx = np.argmin(std)
            res[i, :] = (mar[idx], size[idx], std[idx], mis[idx, 0]) if not cls else (mar[idx], size[idx], std[idx], mis[idx])
    return res

def proc_mar(mar, tol=1e-2, n=30):
    mar_str = []
    app_str = []
    for data in mar:
        if data < 0.9-tol:
            mar_str.append(f'${data:.3f}^-$')
            app_str.append('^-')
        elif data > 0.901+1/n:
            mar_str.append(f'${data:.3f}^+$')
            app_str.append('^+')
        else:
            mar_str.append(f'${data:.3f}$')
            app_str.append('')
    return mar_str, app_str

def proc_std(std, app_str):
    std_str = []
    base = [] if (std[-1] != '') else [std[-1]]
    ora = std[-2]
    for i in range(len(std)):
        if app_str[i] != '':
            std_str.append(f'$\\textit{{{std[i]:.2f}}}{app_str[i]}$')
        else:
            if i < 3:
                base.append(std[i])
            if i in [3, 4]:
                if std[i] < np.min(base):
                    std_str.append(f'$\\makecell{{\\textbf{{{std[i]:.2f}}}\\\\({(np.min(base)-std[i])/(np.min(base)-ora)*100:.1f}\\%)}}$')
                else:
                    std_str.append(f'${std[i]:.2f}(0\\%)$')
            else:
                std_str.append(f'${std[i]:.2f}$')
    return std_str

def proc_size(size, app_str):
    min_size = np.min([size[i] for i in range(7) if ((app_str[i]=='') and (i!=5))])
    size_str = []
    for i in range(7):
        if app_str[i] != '':
            size_str.append(f'$\\textit{{{size[i]:.2f}}}{app_str[i]}$')
        elif (i != 5) and (size[i] <= min_size):
            size_str.append(f'$\\textbf{{{size[i]:.2f}}}$')
        else:
            size_str.append(f'${size[i]:.2f}$')
    return size_str

def to_str(resArr:list, name, n_m, tol=1e-2):
    mar_str, std_str, mis_str, size_str = '', '', '', ''
    for i, (res1, res2) in enumerate(resArr):
        mar1, mar2 = res1[:, 0], res2[:, 0]
        size1, size2 = res1[:, 1], res2[:, 1]
        std1, std2 = res1[:, 2], res2[:, 2]
        mis1, mis2 = res1[:, 3], res2[:, 3]
        mar_str += f"& {name[i]} & ${n_m[i]}$ & " + ' & '.join(proc_mar(mar1, tol)[0]) + ' && ' + ' & '.join(proc_mar(mar2, tol)[0]) + '\\\\\\\\\n'
        app_str1, app_str2 = proc_mar(mar1, tol)[1], proc_mar(mar2, tol)[1]
        std_str += f"& {name[i]} & ${n_m[i]}$ & " + ' & '.join(proc_std(std1, app_str1)) + ' && ' + ' & '.join(proc_std(std2, app_str2)) + '\\\\\\\\\n'
        mis_str += f"& {name[i]} & ${n_m[i]}$ & " + ' & '.join([f"${x:.3f}$" for x in mis1]) + ' && ' + ' & '.join([f"${x:.3f}$" for x in mis2]) + '\\\\\\\\\n'
        size_str += f"& {name[i]} & ${n_m[i]}$ & " + ' & '.join(proc_size(size1, app_str1)) + ' && ' + ' & '.join(proc_size(size2, app_str2)) + '\\\\\\\\\n'
    return mar_str, mis_str, std_str, size_str


paths = ["../SimResult/Real_Crime/P_60_500_1334_0_15.pkl",
         "../SimResult/Real_Protein/P_60_1000_2000_4.pkl",
         "../SimResult/Real_Star/P_60_1048_2446_10_1_1.pkl",
         "../SimResult/Real_Derma/P_30_1000_2000_0.035.pkl",
         "../SimResult/Real_Tissue/P_30_1000_2000_0.07.pkl"
         ]
names = ['CRIME', 'BIO', 'STAR', 'DERMA', 'TISSUE']
n_ms = ['30/500', '30/1000', '30/1000', '30/1000', '30/1000']
is_cls = [False, False, False, True, True]
resDicts = [read_pkl(path) for path in paths]
resArr = [[sum_compare_result(resDicts[i], 'GLCP', 0.01, 30, is_cls[i]), sum_compare_result(resDicts[i], 'SCC', 0.01, 30, is_cls[i])] for i in range(5)]
res_str = to_str(resArr, names, n_ms)

with open('sum_tab.txt', 'w') as f:
    f.write('Std ' + res_str[2])
    f.write('Marginal ' + res_str[0])
    f.write('Size ' + res_str[3])
    f.write('Miscoverage ' + res_str[1])