import numpy as np
import math
import matplotlib.pyplot as plt

def toColorList(C:np.ndarray):
    """
    :param C: c*3
    """
    return [tuple(C[i,:]) for i in range(C.shape[0])]

def continuousColorMap(name:str, a_:float, b_:float):
    """
    :param name: MyColor key
    :param a_: min possible value
    :param b_: max possible value
    :return: a function R -> R^3, x -> (c1,c2,c3)
    """
    band = 1 / (len(MyColor[name]) - 1)
    def fun(x):
        x = (x-a_)/(b_-a_)
        ind = math.floor(x / band)
        if ind == len(MyColor[name]) - 1:
            return MyColor[name][ind]
        weight2 = (x - ind * band) / band
        weight1 = 1 - weight2
        color = np.array(MyColor[name][ind]) * weight1 + \
               np.array(MyColor[name][ind+1]) * weight2
        return tuple(color)
    return fun

MyColor_ = {
    'wlm': np.array([[63,59,114],[97,124,184],[184,168,207],
                     [253,207,158], [223,155,146],[192,107,94]])/255,
    'cts1': np.array([[166,64,54],[234,216,154],
                      [155,180,150],[50,120,138]])/255,
    'cts2': np.array([[82,143,173],[21,29,41],[229,168,75],[174,32,18]])/255,
    'gqj':np.array([[70,120,142],[120,183,201],[246,224,147],
                    [229,139,123],[151,179,25]])/255,
    'fsh':np.array([[231,98,84],[239,138,71],[247,170,88],
                    [255,208,111],[255,230,183],[170,220,224],
                    [114,188,213],[82,143,173],[55,103,149],
                    [30,70,110]])/255,
    'dk2':np.array([[181,212,101],[142,160,199],[215,144,193],
                    [235,143,107],[130,191,166],[250,215,83]])/255,
    'dk1':np.array([[38,70,83],[40,114,113],[41,157,143],
                    [232,197,107],[243,162,97],[230,111,81]])/255,
    'cts3': np.array([[21,29,41],[93,163,157],[229,168,75],[174,32,18]])/255,
    'dark':np.array([[21,29,41],[82,143,173],[174,32,18],[46,139,87],
                     [153,51,250],[135,38,87],[229,168,75],[72,61,139],[255,20,147],[50,205,50]])/255,
    'bright':np.array([[238,130,238],[255,125,64],[165,42,42],[138,43,226],[100,149,237],[50,205,50]])/255
}

MyColor = { i:toColorList(MyColor_[i]) for i in MyColor_.keys()}

def exhibit(name):
    plt.figure(figsize=(8, 4))
    for i in range(len(MyColor[name])):
        plt.scatter([i], [0], s=200, color=MyColor[name][i], label=str(i))
    plt.subplots_adjust(right=0.75)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim((-1, len(MyColor[name])))
    plt.title(f"Color Name: {name}")
    plt.show()

if __name__ == '__main__':
    exhibit('dark')
    exhibit('gqj')