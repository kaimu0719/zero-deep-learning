import os, sys
import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from optimizer.SGD.SGD import SGD
from optimizer.Momentum.Momentum import Momentum
from optimizer.AdaGrad.AdaGrad import AdaGrad
from optimizer.Adam.Adam import Adam

class Trainer:
    """
    ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network # 学習対象のネットワーク
        self.verbose = verbose # ログを表示するか(bool)
        self.x_train = x_train # 訓練データ
        self.t_train = t_train # テストデータ
        self.x_test = x_test # 訓練データの正解ラベル
        self.t_test = t_test # テストデータの正解ラベル
        self.epochs = epochs # 総エポック数
        self.batch_size = mini_batch_size # ミニバッチサイズ
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch # 各エポックで評価に使うサンプル数

        # optimizerを文字列から生成する
        optimizer_class_dict = {
            'sgd':SGD,
            'momentum':Momentum,
            'adagrad':AdaGrad,
            'adam':Adam
        }
        # 小文字化して辞書検索し、**kwargs でパラメータを渡す
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        # 学習ループ用カウンタ
        self.train_size = x_train.shape[0] # 訓練サンプル総数
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1) # 1エポックあたりの更新回数
        self.max_iter = int(epochs * self.iter_per_epoch) # 総イテレーション数
        self.current_iter = 0 # 現在のイテレーション
        self.current_epoch = 0 # 現在のエポック

        # 学習履歴を格納するリスト
        self.train_loss_list = [] # 損失
        self.train_acc_list = [] # 訓練精度
        self.test_acc_list = [] # テスト精度
    
    def train_step(self):
        """
        """
        # ミニバッチをランダム抽出(重複なし)
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 勾配を計算してパラメータを更新
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        # 損失を記録
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print(f'train loss: {loss}')
        
        # エポック終了判定(iter_per_epoch ごと)
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1 # エポックカウンタを進める

            # 評価用サンプルを準備
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            
            # 訓練 & テスト精度を測定
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f'=== epoch: {self.current_epoch}, train acc: {train_acc}, test acc: {test_acc} ===')
        
        # イテレーションを進める
        self.current_iter += 1
    
    # 全学習ループを実行するメソッド
    def train(self):
        for _ in range(self.max_iter):
            self.train_step() # 1ステップずつ進める
        
        # 最終テスト精度を測定して表示
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose:
            print(f'=========== Final Test Accuracy ===========')
            print(f'test acc: {test_acc}')