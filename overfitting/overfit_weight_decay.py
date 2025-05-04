import os, sys
import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.mnist import load_mnist
from network.multi_layer_net_extend import MultiLayerNetExtend
from optimizer.SGD.SGD import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
t_train = t_train[:300]

# weight_decay (荷重減衰)の設定
weight_decay_lambda = 0 # weight decay を使用しない場合
# weight_decay_lambda = 0.1

network = MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[100]*6,
    output_size=10,
    weight_decay_lambda=weight_decay_lambda,
)

optimizer = SGD(lr=0.01)

# 学習ハイパーパラメータ
max_epochs = 201
train_size = x_train.shape[0] # 300
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 epoch あたりのミニバッチ更新回数
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    # ミニバッチの抽出
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配計算 & パラメータ更新
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    # 1 epoch ごとに精度を記録
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'epoch:{epoch_cnt}, train acc:{train_acc}, test acc:{test_acc}')

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 学習曲線の可視化
epochs = np.arange(max_epochs)
plt.plot(epochs, train_acc_list, label='train', marker='o', markevery=10)
plt.plot(epochs, test_acc_list, label='test', marker='s', markevery=10)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
