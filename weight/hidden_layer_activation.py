"""
重みの初期値によって隠れ層のアクティベーション（活性化関数の後の出力データ）が
どのように変化するかを確認するためのコード
"""

import numpy as np
import matplotlib.pyplot as plt
from activation_function import sigmoid, relu, tanh

# 1000行100列の乱数行列を生成
x = np.random.randn(1000, 100)

# 各隠れ層のノードの数
node_num = 100

# 隠れ層が５層
hidden_layer_size = 5

# ここにアクティベーションの結果を格納する
activations = {}

for i in range(hidden_layer_size):
    # 一層目の場合は元の入力データxを使用するが、２層目以降では前の層のアクティベーションを使う
    if i != 0:
        x = activations[i - 1]
    
    # 重みの初期値を標準正規分布に従う乱数で生成
    # w = np.random.randn(node_num, node_num) * 1 # 標準偏差を1にしてみる
    # w = np.random.randn(node_num, node_num) * 0.01 # 標準偏差を0.01にしてみる
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # Xavierの初期値
    # w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num) # Heの初期値

    # 行列積を計算
    z = np.dot(x, w)

    # 活性化関数を適用
    # a = sigmoid(z) # シグモイド関数
    # a = tanh(z) # 双曲線関数
    a = relu(z) # ReLU関数

    # アクティベーションを保存
    activations[i] = a

# ヒストグラムを作成
for i, a in activations.items():
    # グラフを一行で並べて、len(activations)個の列で表示する
    # グラフの配置は１層から順番に並べる
    plt.subplot(1, len(activations), i + 1)

    # タイトルをつける
    plt.title(f'layer {i + 1}')

    # アクティベーションの二次元配列を一列に並べ替えて、
    # データの範囲を30個のビンに分けて
    # ヒストグラムの範囲を0から1に指定して表示
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()