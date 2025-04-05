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
        self.x = None # 入力データ
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
        self.x = x # 入力データを保存
        out = np.dot(x, self.W) + self.b # 線形変換を計算

        return out # 出力値を返す
    
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

        return dx # 入力データの勾配を返す