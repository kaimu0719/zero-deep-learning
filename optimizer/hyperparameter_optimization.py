import sys, os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.mnist import load_mnist
from network.multi_layer_net import MultiLayerNet
from utils.shuffle import shuffle_dataset
from train.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 高速化のため訓練データの削減
x_train = x_train[:500]
t_train = t_train[:500]

# 検証データの分離
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100]*6,
        output_size=10,
        weight_decay_lambda=weight_decay,
    )
    trainer = Trainer(
        network=network,
        x_train=x_train,
        t_train=t_train,
        x_test=x_val,
        t_test=t_val,
        epochs=epocs,
        mini_batch_size=100,
        optimizer='sgd',
        optimizer_param={'lr': lr},
        verbose=False
    )
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

# ハイパーパラメータのランダム探索==============================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 探索したハイパーパラメータの範囲を指定=====================
    weight_decay = 10 ** np.random.uniform(-8, -4) # weight_decay [1e-8, 1e-4]
    lr = 10 ** np.random.uniform(-6, -2) # 学習係数 [1e-6, 1e-2]
    # ====================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print(f'val acc: {val_acc_list[-1]} | lr: {lr}, weight decay: {weight_decay}')
    key = f'lr: {lr}, weight decay: {weight_decay}'
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# グラフの描画==============================================
print("======== Hyper-Parameter Optimization Result ========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print(f'Best-{i+1}(val acc: {val_acc_list[-1]}) | {key}')

    plt.subplot(row_num, col_num, i+1)
    plt.title(f'Best-{i+1}')
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], '--')
    i += 1

    if i >= graph_draw_num:
        break

plt.show()