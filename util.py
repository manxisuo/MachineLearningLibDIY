# encoding: utf-8
from typing import List, Tuple
import numpy as np
from numpy import ndarray


def _normalize(array: ndarray) -> Tuple[float, float]:
    """
    归一化一个数字列表，返回归一化参数。
    """
    rng = array.max() - array.min()
    if rng == 0:
        return np.array([array[0] - 1, 1])
    else:
        return np.array([array.sum() / array.size, rng])


class Normalizer:
    """
    对一组样本的特征值进行归一化的归一化器。
    """
    def __init__(self):
        self.params: List[Tuple[float, float]] = None

    def fit(self, X: ndarray) -> None:
        """
        训练归一化器。
        :param X: 任意行n列矩阵
        """
        self.params = np.apply_along_axis(_normalize, 0, X)

    def transform(self, X: ndarray) -> ndarray:
        """
        使用归一化器进行归一化。
        :param X: 任意行n行矩阵或n维向量
        """
        return (X - self.params[0]) / self.params[1]


if __name__ == '__main__':
    X = np.array([
         [3, 3, 0, 0],
         [1, 3, 0, 1],
         [4, 3, 0, 0],
         [9, 3, 0, 1]])

    # TODO 如果转换数据的值超出了训练集中数据的范围，结果有误
    # TODO 即目前仅能保证对训练集中的数据的转换有效
    X2 = np.array([
        [2, 3, 0, 1]
    ])

    X3 = np.array([2, 3, 0, 1])

    normalizer = Normalizer()
    normalizer.fit(X)

    print(normalizer.transform(X))
    print(normalizer.transform(X2))
    print(normalizer.transform(X3))
