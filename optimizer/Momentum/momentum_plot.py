import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Momentumのインポート
from Momentum import Momentum

def f(x, y):
    """
    optimizerの比較に使用する関数 : 二次関数の形をしている。
    f(x, y) = (x^2 / 20) + (y^2)
    """
    return x**2 / 20.0 + y**2

def df(x, y):
    """
    勾配を計算する関数 : f(x, y)の勾配を計算する。
    """
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)  # 初期位置
params = {}  # パラメータを格納する辞書
params['x'], params['y'] = init_pos[0], init_pos[1]  # 初期位置を設定
grads = {}  # 勾配を格納する辞書
grads['x'], grads['y'] = 0, 0  # 勾配を初期化

optimizer = Momentum(lr=0.1)  # Momentumのインスタンスを作成

x_history = [] # x座標の履歴
y_history = [] # y座標の履歴

for i in range(30):
    x_history.append(params['x'])
    y_history.append(params['y'])

    # 勾配を計算
    grads['x'], grads['y'] = df(params['x'], params['y'])

    # パラメータを更新
    optimizer.update(params, grads)

# 描画用のxy座標範囲を作成
x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# マスク処理（値が大きいところをカットするなど）をするなら任意で実施
# mask = Z > 7
# Z[mask] = 7  # 例: 大きい部分を 7 にクリップ

# 3Dでプロット
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 曲面を描画
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# 勾配降下の経路を3次元空間にプロット
# f(x,y) の値をz座標として持ってくる
z_history = [f(x, y) for x, y in zip(x_history, y_history)]
ax.plot(x_history, y_history, z_history, 'o-', color='red', label='Momentum path')

# 最小値のある点(0,0)を目立たせたい場合
ax.scatter(0, 0, f(0, 0), color='black', s=50, marker='^', label='minimum')

# 軸ラベル等
ax.set_title('Momentum in 3D')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

ax.legend()
plt.show()
