import sys, os
import numpy as np
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from collections import OrderedDict
from gradient.numerical import numerical_gradient
from Backpropagation.AffineLayer import Affine
from Backpropagation.ConvolutionLayer import Convolution
from Backpropagation.PoolingLayer import Pooling
from Backpropagation.SoftmaxWithLossLayer import SoftmaxWithLoss
from Backpropagation.ReLULayer import Relu

class SimpleConvNet:
    """単純なConvNet

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    - input_size : 入力サイズ（MNISTの場合は784）
    - hidden_size_list : 隠れ層のニューロンの数のリスト（e.g）
    - output_size : 出力サイズ（MNISTn場合は10）
    - activation : 'relu' or 'sigmoid'
    - weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """
    def __init__(
            self,
            input_dim=(1, 28, 28),
            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
            hidden_size=100,
            output_size=10,
            weight_init_std=0.01,
        ):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # 4.6
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 158.7
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 重みの初期値
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 重みの初期化
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]
    
    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）
        
        Parameters
        ----------
        - x : 入力データ
        - t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1'], grads['W2'], ...は各層の重み
            grads['b1'], grads['b2'], ...は各層のバイアス
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads[f'W{idx}'] = numerical_gradient(loss_w, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_w, self.params[f'b{idx}'])
        
        return grads
    
    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）

        Parameters
        ----------
        - x : 入力データ
        - t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1'], grads['W2'], ...は各層の重み
            grads['b1'], grads['b2'], ...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]