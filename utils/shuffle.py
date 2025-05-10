import numpy as np

def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    # np.random.permutation(5)の場合
    # 0~4の数字をランダムに入れ替えて、[3, 1, 4, 0, 2]のような値を返す。
    permutation = np.random.permutation(x.shape[0])
    
    # x.ndimは次元数
    if x.ndim == 2: # 2次元(例: (サンプル数, 特徴量))の時は行方向だけ並び替える。
        x = x[permutation,:]
    else: # 画像のように4次元(例: (サンプル数, チャンネル, 高さ, 幅))の時は、先頭次元だけを並び替え、残りはそのまま持っておく
        x[permutation,:,:,:]
    
    # 教師データを同じ順序で並び替える。
    t = t[permutation]

    return x, t