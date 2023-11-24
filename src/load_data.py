import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# sklearnデータセットに収録されたiris(アヤメ)のデータセットをロードしてデータフレームを作成
def load_iris_data():
    data = load_iris()
    x = pd.DataFrame(data["data"],columns=data["feature_names"])
    y = pd.DataFrame(data["target"],columns=["target"])
    return x, y

# 手書き文字のデータセットをダウンロードして、実験用データを準備 (70000枚のうち7000枚を利用)
def load_mnist_data():
    data = fetch_openml('mnist_784', version=1)
    _x = np.array(data['data'].astype(np.float32))
    _y = np.array(data['target'].astype(np.int32))
    _, x, _, y = train_test_split(_x, _y, test_size=0.1, random_state=1, stratify=_y)
    return x, y

# Fashion-MNISTデータセットをダウンロードして、実験用データを準備 (70000枚のうち7000枚を利用)
def load_fashion_mnist_data():
    data = fetch_openml('Fashion-MNIST')
    _x = np.array(data['data'].astype(np.float32))
    _y = np.array(data['target'].astype(np.int32))
    _, x, _, y = train_test_split(_x, _y, test_size=0.1, random_state=1, stratify=_y) 
    return x, y

def reshape_image(x, y=None):
    """画像データのサイズ変更

    Args:
        x (array_like): input
        y (array_like, optional): output. Defaults to None.

    Returns:
        x, y: reshaped data
        
    Samples:
        ```python
        >>> x, y = reshape_image(x)
        >>> print(x)
        # shape = (T_i, 28, 28)
        >>> print(y)
        # shape = (T_i, 10, 1) (Onehot-vector)
        ```
    """
    if y is None:
        x, y = x
    x = x.reshape(-1, 28, 28)
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    return x / 255.0, y

def dataset():
    """一括処理のためにデータセットの辞書を作成

    Returns:
        dict: keys=("iris", "mnist", "fashion-mnist")
    """
    dataset= {
        'iris': load_iris_data(),
        'mnist': reshape_image(load_mnist_data()),
        'fashion-mnist': reshape_image(load_fashion_mnist_data())
    }
    return dataset


if __name__ == "__main__":
    ds = dataset()
    print(ds["mnist"])
