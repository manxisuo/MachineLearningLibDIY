from typing import List, Tuple


def _normalize(array: List[float]) -> Tuple[float, float]:
    """
    归一化一个数字列表，返回归一化参数。
    """
    _min, _max, s, size = array[0], array[0], 0, len(array)
    for i in array:
        if i < _min:
            _min = i
        elif i > _max:
            _max = i
        s += i

    avg, rng = s / size, _max - _min

    if rng == 0:
        return array[0] - 1, 1
    else:
        return avg, rng


class Normalizer:
    """
    对一组样本的特征值进行归一化的归一化器。
    """
    def __init__(self):
        self.params: List[Tuple[float, float]] = None

    def fit(self, xs: List[List[float]]):
        """训练归一化器。"""
        n = len(xs[0])
        self.params = [None] * n

        for i in range(n):
            self.params[i] = _normalize([x[i] for x in xs])

    def transform(self, xs: List[List[float]]):
        """使用归一化器进行归一化。"""
        n = len(xs[0])
        normalize_xs = []
        for x in xs:
            normalize_x = []
            for i in range(n):
                avg, rng = self.params[i]
                normalize_x.append((x[i] - avg) / rng)
            normalize_xs.append(normalize_x)
        return normalize_xs


if __name__ == '__main__':
    xs = [
        [3, 3, 0, 0],
        [1, 3, 0, 1],
        [4, 3, 0, 0],
        [9, 3, 0, 1]
    ]

    normalizer = Normalizer()
    normalizer.fit(xs)
    print(normalizer.transform(xs))
