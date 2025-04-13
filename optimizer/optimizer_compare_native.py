import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# optimizerのインポート
from AdaGrad.AdaGrad import AdaGrad
from Adam.Adam import Adam
from Momentum.Momentum import Momentum
from SGD.SGD import SGD

def f(x, y):
    """
    optimizerの比較に使用する関数 : 二次関数の形をしている。
    f(x, y) = (x^2 / 20) + (y^2)

    Parameters
    ----------
    - x : x座標
    - y : y座標

    Returns
    -------
    - f(x, y) : 二次関数の値
    """
    return x**2 / 20.0 + y**2

def df(x, y):
    """
    勾配を計算する関数 : f(x, y)の勾配を計算する。

    Parameters
    ----------
    - x : x座標
    - y : y座標

    Returns
    -------
    - df(x, y) : 勾配の値
    """
    return x / 10.0, 2.0*y


init_pos = (-7.0, 2.0) # 初期位置
params = {} # パラメータを格納する辞書
params['x'], params['y'] = init_pos[0], init_pos[1] # 初期位置を設定
grads = {} # 勾配を格納する辞書
grads['x'], grads['y'] = 0, 0 # 勾配を初期化

# 各optimizerのインスタンスを作成
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1 # subplotのインデックス

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []

    # 各optimizerを比較する際に、すべて同じ初期位置からスタートするため、paramsを初期化 
    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()
