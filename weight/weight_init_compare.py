import os, sys

# 現在のファイル(optimizer_compare_mnist.py)があるディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# その親ディレクトリを取得
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from data.mnist import load_mnist
from utils.smooth import smooth_curve
from network.multi_layer_net import MultiLayerNet
from optimizer.SGD.SGD import SGD

# MNISTデータの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0] # 訓練データの数
batch_size = 128 # 一回の訓練で使うデータの数
max_iterations = 2000 # 訓練の総回数

weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100]*4,
        output_size=10,
        weight_init_std=weight_type,
    )
    train_loss[key] = [] # 各重み初期化の損失を格納するリスト

# 訓練の開始
for i in range(max_iterations):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():

        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# グラフの描画
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0, 2.5)
plt.legend()
plt.show()