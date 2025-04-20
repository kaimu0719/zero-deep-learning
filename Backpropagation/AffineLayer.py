import numpy as np

class Affine:
    """
    Affineレイヤー : このレイヤーは、線形変換を行うためのレイヤーであり、入力データに対して重み行列とバイアスベクトルを使用して変換を行う。
    """
    def __init__(self, W, b):
        """
        Affineレイヤーの初期化メソッド。

        Parameters
        ----------
        - W : 重み行列（NumPy配列）
        - b : バイアスベクトル（NumPy配列）
        """
        self.W = W # 重み行列
        self.b = b # バイアスベクトル
        self.original_x_shape = None # 入力データ
        self.dW = None # 重みの勾配
        self.db = None # バイアスの勾配

    def forward(self, x):
        """
        順伝播（forward propagation） : 入力された値（x）に対して線形変換を適用し、その結果を返す。

        Parameters
        ----------
        - x : 入力値（NumPy配列）

        Returns
        -------
        - out : 線形変換を適用した出力値
        """
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out
    
    def backward(self, dout):
        """
        逆伝播（backward propagation） : 順伝播で計算した出力を使用して、勾配（dout）を計算する。

        Parameters
        ----------
        - dout : 勾配（NumPy配列）

        Returns
        -------
        - dx : 入力データの勾配
        """
        dx = np.dot(dout, self.W.T) # 入力データの勾配を計算
        self.dW = np.dot(self.x.T, dout) # 重みの勾配を計算
        self.db = np.sum(dout, axis=0) # バイアスの勾配を計算

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx # 入力データの勾配を返す