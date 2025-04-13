import sys, os
import numpy as np
from collections import OrderedDict

sys.path.append(os.pardir)

from Backpropagation.AffineLayer import Affine
from Backpropagation.ReLULayer import Relu
from Backpropagation.SigmoidLayer import Sigmoid
from Backpropagation.SoftmaxWithLossLayer import SoftmaxWithLoss

from gradient.numerical import numerical_gradient

class MultiLayerNet:
    """
    多層ニューラルネットワーク

    Parameters
    ----------
    - input_size : 入力サイズ（MNISTの場合は784）
    - hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    - output_size : 出力サイズ（MNISTの場合は10）
    - activation : 活性化関数（'relu' or 'sigmoid'）
    - weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    - weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size # 入力サイズ
        self.output_size = output_size # 出力サイズ
        self.hidden_size_list = hidden_size_list # 隠れ層のサイズリスト
        self.hidden_layer_num = len(hidden_size_list) # 隠れ層の数
        self.weight_decay_lambda = weight_decay_lambda # 重みの初期化方法
        self.params = {} # パラメータ（重みとバイアス）を格納

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}

        # 隠れ層の数だけレイヤを追加
        self.layers = OrderedDict()

        # 隠れ層の数だけAffineレイヤーと活性化関数レイヤーを追加
        for idx in range(1, self.hidden_layer_num+1):
            # Affineレイヤを追加
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 活性化関数レイヤを追加
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        
        # 出力層のAffineレイヤを追加
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        # 出力層の活性化関数レイヤを追加
        self.last_layer = SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std):
        """
        重みの初期値設定

        Parameters
        ----------
        - weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        # 隠れ層の数を取得
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        # 隠れ層の数だけ重みとバイアスを初期化
        for idx in range(1, len(all_size_list)):
            # 重みの初期化
            scale = weight_init_std

            # 重みの初期化方法を指定
            if str(weight_init_std).lower() in ('relu', 'he'):
                # Reluを使う場合に推奨される初期値
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
    
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                # Sigmoidを使う場合に推奨される初期値
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            
            # 重みとバイアスを初期化
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
    
    def predict(self, x):
        """
        順伝播（forward propagation） : 入力データをネットワークに通して予測値を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        """
        損失関数の計算 : 順伝播を行い、損失（loss）を計算する。
        交差エントロピー誤差を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 教師データ（one-hot vector）

        Returns
        -------
        - loss : 交差エントロピー誤差
        """
        # 順伝播を行い、予測値を計算
        y = self.predict(x)

        weight_decay = 0 # L2正則化項を初期化
        for idx in range(1, self.hidden_layer_num + 2): # 隠れ層の数+出力層の数

            # 隠れ層の数だけ重みを取得
            W = self.params['W' + str(idx)]

            # L2正則化項を計算
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        
        # 損失を計算
        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        """
        精度の計算 : 順伝播を行い、予測値を計算し、正解率を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 教師データ（one-hot vector）

        Returns
        -------
        - accuracy : 正解率
        """
        # 順伝播を行い、予測値を計算
        y = self.predict(x)

        # 予測値をone-hot vectorからラベルに変換
        y = np.argmax(y, axis=1)

        # 教師データがone-hot vectorの場合、正解ラベルのインデックスに変換
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        # 正解率を計算
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
    
    def numerical_gradient(self, x, t):
        """
        数値微分を用いて勾配を計算するメソッド

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 教師データ（one-hot vector）

        Returns
        -------
        - grads : 勾配（辞書型）
            grads['W1'], grads['W2'], ...各層の重み
            grads['b1'], grads['b2'], ...各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads
    
    def gradient(self, x, t):
        """
        勾配を求める（誤差逆伝播法）

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 教師データ（one-hot vector）

        Returns
        -------
        - grads : 勾配（辞書型）
            grads['W1'], grads['W2'], ...各層の重み
            grads['b1'], grads['b2'], ...各層のバイアス
        """
        # 順伝播を行い、損失を計算
        self.loss(x, t)

        # 逆伝播を行い、勾配を計算
        dout = 1
        dout = self.last_layer.backward(dout)

        # 隠れ層の数だけ逆伝播を行う
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        

        # 各Affineレイヤで既に計算された勾配(dW, db)を取得し、
        # Weight Decay（L2正則化）項を加算したうえで辞書gradsに格納している。
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        
        return grads