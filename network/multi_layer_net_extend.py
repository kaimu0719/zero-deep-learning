import sys, os
import numpy as np
from collections import OrderedDict

sys.path.append(os.pardir)

from Backpropagation.AffineLayer import Affine
from Backpropagation.ReLULayer import Relu
from Backpropagation.SigmoidLayer import Sigmoid
from Backpropagation.SoftmaxWithLossLayer import SoftmaxWithLoss
from Backpropagation.BatchNormalization import BatchNormalization

from gradient.numerical import numerical_gradient

class MultiLayerNetExtend:
    """
    全結合による多層ニューラルネットワークの拡張版クラス
    - Weight Decay(L2正規化)
    - Batch Normalization
    を1クラスで扱えるようにした実験用ネットワーク
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_batchnorm=False):
        """
        ネットワークの"骨格"を組み立てるコンストラクタ

        Parameters
        ----------
        - input_size : int
            入力層ユニット数(例: MNISTは784)
        - hidden_size_list : list[int]
            隠れ層ごとのユニット数(ニューラルネットワークのニューロン1つ)を並べたリスト
            例) [100, 100, 100] → 隠れ層1,2,3のユニット数がそれぞれ100
        - output_size : int
            出力層ユニット数(例: MNISTは10)
        - activation : {'relu', 'sigmoid'}
            隠れ層で使う活性化関数
        - weight_init_std : {'relu', 'he', 'sigmoid', 'xavier'} or float
            重み初期化方法または標準偏差の数値を直接指定
        - weight_decay_lambda : float
            L2正規化係数(0 なら Weight Decay 無効)
        - use_batchnorm : bool
            True で各隠れ層直後に Batch Normalization を挿入する
        """

        # 引数をプロパティへ保存
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list) # 隠れ層の数
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm

        # 全レイヤのパラメータ(W, b, γ, β)を格納する dict
        # keys : 'W1', 'b1', 'gamma1', 'beta1', ... と並ぶ
        self.params = {}

        # 重みの初期化
        # 具体的な生成ロジックは private メソッド __init_weight() へ分離
        self.__init_weight(weight_init_std)

        # レイヤオブジェクトを順番に構築
        # OrderedDict にすることで forward は定義順
        # backward は reversed() で逆順に簡単に回せる
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()

        # 隠れ層ブロックを 1 ~ hidden_layer_num までループ生成
        for idx in range(1, self.hidden_layer_num+1):
            # Affine(全結合)レイヤを追加
            self.layers[f'Affine{idx}'] = Affine(
                self.params[f'W{idx}'],
                self.params[f'b{idx}']
            )

            # BatchNorm(任意)
            if self.use_batchnorm:
                # γ, β も params に登録して学習対象にする
                self.params[f'gamma{idx}'] = np.ones(hidden_size_list[idx-1])
                self.params[f'beta{idx}'] = np.zeros(hidden_size_list[idx-1])
                self.layers[f'BatchNorm{idx}'] = BatchNormalization(self.params[f'gamma{idx}'], self.params[f'beta{idx}'])
            
            # 活性化レイヤ
            self.layers[f'Activation_function{idx}'] = activation_layer[activation]()
        
        # 出力層
        idx = self.hidden_layer_num + 1
        self.layers[f'Affine{idx}'] = Affine(
            self.params[f'W{idx}'],
            self.params[f'b{idx}']
        )

        # 損失関数レイヤ(Softmax+交差エントロピー)
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

        # ----- レイヤー構成を1本のリストにまとめる -----
        # [入力層, 隠れ層1, ..., 隠れ層N, 出力層]という順序リストができる。
        # ↓
        # 1) ループで(前の層のユニット数, 次の層のユニット数)のペアを次々に取り出せるため、
        #    重み行列 W_i(shape = (前の層, 次の層))とバイアス b_i(shape = (次,))を自動生成しやすい。
        # 2) 隠れ層を増減する時は hidden_size_list だけを変えればよく、
        #    それ以外の初期化コードを触らずにネットワーク構造を変更できる。
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        # ----- 重み(W1...WL)とバイアス(b1...bL)をまとめて初期化する -----
        # ループの考え方
        # idx = 1 → 入力層(0) → 隠れ層1
        # idx = 2 → 隠れ層1 → 隠れ層2
        # |
        # idx = L → 隠れ層(N) → 出力層
        #
        # all_size_list[idx-1] ← 前の層ユニット数
        # all_size_list[idx] ← 次の層ユニット数
        # L = len(all_size_list) - 1 # 隠れ層の数
        for idx in range(1, len(all_size_list)): # idxは 1,2,...,L
            # 初期乱数の標準偏差の係数を決定する
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                # Relu 活性化関数を使用 : He 初期化 (σ = √2 / all_size_list[idx-1])
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                # Sigmoid 活性化関数を使用 : Xavier 初期化 (σ = 1 / √all_size_list[idx-1])
                scale = 1.0 / np.sqrt(all_size_list[idx-1])
            
            # 重み行列とバイアスベクトルを生成
            # W_idx : (前の層ユニット数, 次の層ユニット数)行列を「scale × 標準正規乱数」で生成
            self.params[f'W{idx}'] = scale * np.random.randn(
                all_size_list[idx-1], # 行 : 前の層ユニット数
                all_size_list[idx] # 列 : 次の層ユニット数
            )
            # b_idx : (次の層ユニット数)要素全て0で初期化
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])
        
    def predict(self, x , train_flg=False):
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            
        return x
        
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        
        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = numerical_gradient(loss_W, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_W, self.params[f'b{idx}'])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads[f'gamma{idx}'] = numerical_gradient(loss_W, self.params[f'gamma{idx}'])
                grads[f'beta{idx}'] = numerical_gradient(loss_W, self.params[f'beta{idx}'])
        
        return grads
    
    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + self.weight_decay_lambda * self.params[f'W{idx}']
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads[f'gamma{idx}'] = self.layers[f'BatchNorm{idx}'].dgamma
                grads[f'beta{idx}'] = self.layers[f'BatchNorm{idx}'].dbeta
        
        return grads

