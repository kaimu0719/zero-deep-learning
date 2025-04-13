import numpy as np

def softmax(x: np.ndarray):
    """
    softmax関数 : 入力された値をsoftmax関数に通して確率分布に変換。

    Parameters
    ----------
    - x : 入力値（Numpy配列）

    Returns
    -------
    - y : softmax関数を通した出力値（Numpy配列）
    """
    c = np.max(x) # オーバーフロー対策
    exp_x = np.exp(x - c) # exp(x)の計算
    sum_exp_x = np.sum(exp_x) # exp(x)の合計
    y = exp_x / sum_exp_x # softmaxの計算

    return y

def cross_entropy_error(y, t):
    """
    交差エントロピー誤差 : 予測値(y)と正解ラベル(t)の交差エントロピー誤差を計算。

    Parameters
    ----------
    - y : 予測値（softmaxの出力）
    - t : 正解ラベル（one-hot vector）

    Returns
    -------
    - loss : 交差エントロピー誤差
    """
    # yがone-hot vectorの場合、正解ラベルのインデックスを取得
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # バッチサイズ
    delta = 1e-7 # 数値安定性のための微小値

    # 交差エントロピー誤差の計算
    """
    例:
    - バッチサイズが3の場合
    np.arange(batch_size)
    # → [0, 1, 2]

    - y[np.arange(batch_size), t] は何をしているのか？
    →「各データごとの予測値の中から、正解クラスの予測確率だけを取り出す」
    例えば、3クラス問題でバッチサイズが3のとき
    y = [[0.1, 0.7, 0.2],   # 1つ目のデータの予測値
        [0.3, 0.3, 0.4],   # 2つ目のデータの予測値
        [0.05, 0.9, 0.05]] # 3つ目のデータの予測値
    正解クラスのインデックス（数字のラベル）t は
    t = [1, 2, 1] # 正解クラス：1つ目は1番、2つ目は2番、3つ目は1番
    このとき、
    y[np.arange(batch_size), t] は
    y[[0, 1, 2], [1, 2, 1]]を取り出すことになるので
    = [0.7, 0.4, 0.9] という結果になる。

    - np.log(...) + deltaとは何か？
    log(0)は無限大になるので、数値安定性のために微小値deltaを足している。

    - 最後にに全データを足し合わせて、平均を取る
    np.sum(...)で全データの交差エントロピー誤差を足し合わせる。
    batch_sizeで割ることで、平均をとる。
    -符号に変えるのは、交差エントロピー誤差は「小さいほど良い」ので、符号を変える。
    """
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

class SoftmaxWithLoss:
    """
    Softmax with Loss Layer : softmax関数と交差エントロピー誤差を組み合わせたレイヤ
    """
    def __init__(self):
        """
        Softmax with Loss Layerの初期化メソッド。
        """
        self.loss = None # 損失値
        self.y = None # softmaxの出力
        self.t = None # 教師データ（one-hot vector）
    
    def forward(self, x, t):
        """
        順伝播（forward propagation） : 入力された値（x）をsoftmax関数に通して確率分布に変換し、交差エントロピー誤差を計算。

        Parameters
        ----------
        - x : 入力値（Numpy配列）
        - t : 教師データ（one-hot vector）

        Returns
        -------
        - loss : 交差エントロピー誤差
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        """
        逆伝播（backward propagation）: 出力側から伝わってきた勾配（dout）を、それぞれの入力値に伝える。

        Parameters
        ----------
        - dout : 出力側から伝わってきた勾配
    
        Returns
        -------
        - dx : softmaxの出力に対する勾配
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx