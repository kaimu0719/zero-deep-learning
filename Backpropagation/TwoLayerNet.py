import numpy as np
from collections import OrderedDict
from Backpropagation.AffineLayer import Affine
from Backpropagation.ReLULayer import Relu
from Backpropagation.SoftmaxWithLossLayer import SoftmaxWithLoss

from tqdm import tqdm

def numerical_gradient(f, x):
    h = 1e-4 # 微小値
    grad = np.zeros_like(x) # 勾配の初期化

    for idx in tqdm(np.ndindex(x.shape), total=x.size, desc="Calculating gradients"):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)

        # 勾配の計算
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 元の値に戻す
        x[idx] = tmp_val

    return grad

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        TwoLayerNetの初期化メソッド。

        Parameters
        ----------
        - input_size : 入力層のサイズ
        - hidden_size : 隠れ層のサイズ
        - output_size : 出力層のサイズ
        - weight_init_std : 重みの初期化に使用する標準偏差
        """

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの初期化
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()

        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        """
        順伝播（forward propagation） : 入力データ（x）をネットワークに通して予測を行う。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）

        Returns
        -------
        - y : ネットワークの出力（予測結果）
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        """
        損失関数の計算 : 順伝播を行い、損失を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 正解ラベル（NumPy配列）

        Returns
        -------
        - loss : 損失値
        """
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)

        return loss
    
    def accuracy(self, x, t):
        """
        正解率の計算 : 順伝播を行い、正解率を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 正解ラベル（NumPy配列）
        
        Returns
        -------
        - accuracy : 正解率
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim !=1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
    
    def numerical_gradient(self, x, t):
        """
        数値勾配の計算 : 重みパラメータに対する勾配を数値微分によって求める。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 正解ラベル（NumPy配列）

        Returns
        -------
        - grads : 各パラメータの勾配（辞書形式）
        """
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        """
        勾配の計算 : 順伝播を行い、逆伝播を通じて勾配を計算する。

        Parameters
        ----------
        - x : 入力データ（NumPy配列）
        - t : 正解ラベル（NumPy配列）

        Returns
        -------
        - grads : 各パラメータの勾配（辞書形式）
        """
        # 順伝播
        self.loss(x, t)

        # 逆伝播
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配の取得
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads