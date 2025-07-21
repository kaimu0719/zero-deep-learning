import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    # 畳み込み演算の出力サイズ（高さと幅）を計算
    # http://qiita.com/DeepTama/items/379cac9a73c2aed7a082#%E2%85%B0-%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E6%BC%94%E7%AE%97
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # input_dataのshapeは(N, C, H, W)→N:バッチサイズ, C:チャンネル数, H:高さ, W:幅
    # [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant' の意味は
    # バッチ軸(N)にはpaddingしない(0,0)
    # チャンネル軸(C)にもpaddingしない
    # 高さ(H)にpadを上下に追加
    """
    元の画像：
    [[[1 2 3]
    [4 5 6]
    [7 8 9]]]

    パディング後：
    [[[0 0 0 0 0]
    [0 1 2 3 0]
    [0 4 5 6 0]
    [0 7 8 9 0]
    [0 0 0 0 0]]]
    """
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col