# encoding: utf-8
from typing import List
from time import time


class History:
    """表示模型训练过程的历史记录"""
    def __init__(self, loss_list, consuming_time):
        self.loss_list = loss_list  # 每次迭代后的损失函数的值的列表
        self.consuming_time = consuming_time  # 训练耗时


def _hypothesis(theta: List[float], x: List[float]) -> float:
    """假设函数。"""
    p = 0
    for theta_i, x_i in zip(theta, x):
        p += theta_i * x_i
    return p


def _loss(theta: List[float], xs: List[List[float]], ys: List[float]) -> float:
    """损失函数。"""
    m = len(xs)
    r = 0
    for x, y in zip(xs, ys):
        r += (_hypothesis(theta, x) - y) ** 2
    r /= (2 * m)
    return r


def _gradient_of_loss(theta: List[float], xs: List[List[float]], ys: List[float]) -> List[float]:
    """损失函数对theta的梯度。"""
    m = len(xs)
    n = len(xs[0])  # 特征数，包括x0=1这个特殊的特征
    gradients = [0] * n
    for x, y in zip(xs, ys):
        diff = (_hypothesis(theta, x) - y)
        for j in range(n):
            gradients[j] += diff * x[j]
    for j in range(n):
        gradients[j] /= m
    return gradients


class LinearRegressionModel:
    """一元线性回归模型。"""
    def __init__(self):
        self.xs = None  # 训练样本的特征值的列表。
        self.ys = None  # 训练样本的Y值。
        self.theta = None  # 模型参数

    def fit(self, xs: List[List[float]], ys: List[float],
            alpha=0.01, num_iteration: int = None, epsilon: float = None,
            show_process=False, save_history=False) -> History:
        """
        训练模型。

        :param ys: 训练样本的特征值的列表。
        :param xs: 训练样本的Y值。
        :param alpha: 学习速率。
        :param num_iteration: 终止条件：迭代次数。设为0或负数时忽略。
        :param epsilon: 终止条件：连续两次迭代中，损失函数的下降量。设为0或负数时忽略。
        :param show_process: 是否显示迭代过程。
        :param save_history: 是否返回迭代过程中的记录信息。
        """
        start = time()
        self.xs = list(map(lambda x: [1.0] + x, xs))  # 在每个x的前面添加一个x0=1，对应theta0
        self.ys = ys
        n = len(self.xs[0])  # 特征数，包括x0=1这个特殊的特征
        theta = [0] * n  # 模型参数初始化
        loss_list = []  # 每次迭代后的损失的列表
        k = 0  # 迭代次数
        previous_loss = _loss(theta, self.xs, self.ys)  # 前一次的损失函数值
        satisfied = False

        while not satisfied:
            k += 1
            theta = self._iter_theta(theta, alpha, n)

            # 检查迭代次数条件是否满足
            if num_iteration and 0 < num_iteration < k:
                satisfied = True

            # 此处是为了减少loss的计算次数
            if (epsilon and 0 < epsilon) or show_process or save_history:
                loss = _loss(theta, self.xs, self.ys)

            # 检查损失下降条件是否满足
            if epsilon and 0 < epsilon:
                if abs(loss - previous_loss) <= epsilon:
                    satisfied = True
                previous_loss = loss

            if show_process:
                print(f'epoch: {k}, loss: {loss}, theta: {theta}')

            if save_history:
                loss_list.append(loss)

        self.theta = theta

        return History(loss_list, time() - start)

    def evaluate(self, xs_test: List[List[float]], ys_test: List[float]) -> float:
        xs_test = list(map(lambda x: [1.0] + x, xs_test))
        loss = _loss(self.theta, xs_test, ys_test)
        return loss

    def predict(self, x: List[float]) -> float:
        """使用模型预测。"""
        x = [1.0] + x
        if self.theta:
            return _hypothesis(self.theta, x)
        else:
            return None

    def _iter_theta(self, theta: List[float], alpha: float, n: int) -> List[float]:
        """
        对theta进行一次迭代。
        即进行一次梯度下降。
        """
        r = _gradient_of_loss(theta, self.xs, self.ys)
        for j in range(n):
            r[j] = theta[j] - alpha * r[j]
        return r
