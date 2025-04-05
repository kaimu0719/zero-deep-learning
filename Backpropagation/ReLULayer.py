class Relu:
    """
    ReLU（Rectified Linear Unit）レイヤ : このレイヤは、入力値が0より大きい場合はそのまま出力し、0以下の場合は0を出力。
    """
    def __init__(self):
        """
        ReLU（Rectified Linear Unit）レイヤーの初期化メソッド。
        """
        self.mask = None # 順伝播で使用するマスクを初期化

    def forward(self, x):
        """
        順伝播（forward propagation） : 入力された値（x）に対してReLUを適用し、その結果を返す。

        Parameters
        ----------
        - x : 入力値（NumPy配列）

        Returns
        -------
        - out : ReLUを適用した出力値
        """
        self.mask = (x <= 0) # xが0以下の位置をTrueにするマスクを作成
        out = x.copy() # 入力xのコピーを作成
        out[self.mask] = 0 # マスクがTrueの位置を0にする

        return out # 出力値を返す

    def backward(self, dout):
        """
        逆伝播（backward propagation） : 順伝播で作成したマスクを使用して、勾配（dout）を計算する。

        Parameters
        ----------
        - dout : 勾配（NumPy配列）
        
        Returns
        -------
        - dx : ReLUの勾配
        """
        dout[self.mask] = 0 # マスクがTrueの位置を0にする
        dx = dout

        return dx # 勾配を返す
        