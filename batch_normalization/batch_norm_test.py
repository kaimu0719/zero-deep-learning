import os, sys
import numpy as np
import matplotlib.pyplot as plt
# 現在のファイル(optimizer_compare_mnist.py)があるディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# その親ディレクトリを取得
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.mnist import load_mnist
from network.multi_layer_net_extend import MultiLayerNetExtend
from optimizer.SGD.SGD import SGD
from optimizer.Adam.Adam import Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 学習データを削減
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100]*5,
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True,
    )
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100]*5,
        output_size=10,
        weight_init_std=weight_init_std,
    )
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        # ミニバッチを作る
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 2つのネットワークで誤差逆伝播を行う
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)

            # ----★ ここに追加 ★----
            if i == 0 and _network is bn_network:     # 1回目の更新だけ見ればOK
                print("W1 grad  :", np.linalg.norm(grads['W1']))
                print("gamma1 grad:", np.linalg.norm(grads.get('gamma1', 0)))
                print("Affine1.dW inside layer:", np.linalg.norm(_network.layers['Affine1'].dW))

            optimizer.update(_network.params, grads)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print(f'epoch: {epoch_cnt} | train acc: {train_acc} - bn train acc: {bn_train_acc}')

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
        
    return train_acc_list, bn_train_acc_list

# グラフの描画
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print(f'================= {i+1}/16 =================')
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, i+1)
    plt.title(f'W: {w}')
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)
    
    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel('epochs')
    plt.legend(loc='lower right')

plt.show()