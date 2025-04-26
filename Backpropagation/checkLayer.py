"""
自作したレイヤーのbackward実装が正しいかを数値勾配で検証するためのスクリプト
"""

import os, sys
import numpy as np

# 現在のファイル(optimizer_compare_mnist.py)があるディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# その親ディレクトリを取得
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from gradient.numerical import numerical_gradient



def layer_grad_check(layer, x, dout):
    """
    forward → backwardで得たdxと、
    同じdoutを使って定義した擬似損失Σ(out * dout)の数値勾配を比較して相対誤差を返す。

    Parameters
    ----------
    - layer : レイヤーのインスタンス
    - x     : 入力データ
    - dout  : 出力の勾配

    Returns
    -------
    - diff : 相対誤差
    """
    # forwardとbackwardを実行してdxを取得する
    layer.forward(x)
    dx = layer.backward(dout)

    # 「same loss = Σ(out * dout)」で数値勾配を取る
    f = lambda inp: np.sum(layer.forward(inp) * dout)
    grad_num = numerical_gradient(f, x)

    diff = np.linalg.norm(grad_num - dx) / (np.linalg.norm(grad_num) + 1e-7)
    return diff

# Affineレイヤのテスト
from Backpropagation.AffineLayer import Affine

# Affineレイヤの初期化
W = np.random.randn(784, 50)
b = np.zeros(50)
aff = Affine(W, b)
x  = np.random.randn(3, 784)
dout = np.random.randn(3, 50)
print('Affine dX diff:', layer_grad_check(aff, x, dout))

# Sigmoidレイヤのテスト
import numpy as np
from Backpropagation.SigmoidLayer import Sigmoid
from gradient.numerical import numerical_gradient

sig  = Sigmoid()
x    = np.random.randn(4, 7)
dout = np.random.randn(4, 7)
print('Sigmoid dX diff:', layer_grad_check(sig, x, dout))

# Reluレイヤのテスト
from Backpropagation.ReLULayer import Relu

relu = Relu()
x    = np.random.randn(5, 9)
dout = np.random.randn(5, 9)
print('ReLU dX diff:', layer_grad_check(relu, x, dout))

# SoftmaxWithLossレイヤのテスト
from Backpropagation.SoftmaxWithLossLayer import SoftmaxWithLoss

def softmax_grad_check(x, t):
    """
    SoftmaxWithLossレイヤの逆伝播勾配を数値微分と比較して検証する。

    Parameters
    ----------
    - x : np.ndarray
        形状（N, C）のスコア列（バッチ N, クラス C）。
    - t : np.ndarray
        教師ラベル。one-hot（N, C）または整数ラベル（N,）を想定。
    
    Returns
    -------
    - diff : float
        実装勾配と数値勾配の相対誤差（0に近いほど正確）。
    """
    layer = SoftmaxWithLoss() # インスタンス生成
    layer.forward(x, t)
    dx = layer.backward()

    # 損失を返す無名関数を定義
    f = lambda inp: layer.forward(inp, t)
    grad_num = numerical_gradient(f, x)

    diff = np.linalg.norm(grad_num - dx) / (np.linalg.norm(grad_num) + 1e-7)
    return diff

batch, classes = 4, 10
x = np.random.randn(batch, classes)

# --- ① one‑hot ---
labels = np.random.randint(0, classes, size=batch)
t_onehot = np.eye(classes)[labels]
print('SoftmaxWithLoss (one‑hot) dX diff:', softmax_grad_check(x, t_onehot))

# --- ② label ---
print('SoftmaxWithLoss (label)   dX diff:', softmax_grad_check(x, labels))
