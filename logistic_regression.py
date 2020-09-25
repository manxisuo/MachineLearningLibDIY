# encoding: utf-8
from typing import Tuple
import numpy as np
from numpy import ndarray
from algo import bgd, History


def _hypothesis(theta: ndarray, x: ndarray) -> float:
    """假设函数。"""
    return 1 / (1 + np.exp(-x @ theta))


def _loss(theta: ndarray, X: ndarray, y: ndarray) -> float:
    """损失函数。"""
    m = X.shape[0]
    predictions = 1 / (1 + np.exp(-X @ theta))
    temp = y @ (np.log(predictions)) + (1 - y) @ (np.log(1 - predictions))
    return temp / (-m)


def _gradient_of_loss(theta: ndarray, X: ndarray, y: ndarray) -> ndarray:
    """损失函数对theta的梯度。"""
    m = X.shape[0]
    predictions = 1 / (1 + np.exp(-X @ theta))  # 逻辑回归的假设函数的取值向量（m*1维）
    diff = predictions - y
    gradients = diff @ X / m
    return gradients


class LogisticRegressionModel:
    """一元线性回归模型。"""
    def __init__(self):
        self.theta = None  # 模型参数

    def fit(self, X: ndarray, y: ndarray,
            alpha=0.01, num_iteration: int = None, epsilon: float = None,
            show_process=False, save_history=False) -> History:
        """
        训练模型。

        :param X: 训练样本的特征值的列表。
        :param y: 训练样本的Y值。
        :param alpha: 学习速率。
        :param num_iteration: 终止条件：迭代次数。设为0或负数时忽略。
        :param epsilon: 终止条件：连续两次迭代中，损失函数的下降量。设为0或负数时忽略。
        :param show_process: 是否显示迭代过程。
        :param save_history: 是否返回迭代过程中的记录信息。
        """
        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)) * 1.0, X])  # 在每个x的前面添加一个x0=1，对应theta0

        theta, history = bgd(X, y, _loss, _gradient_of_loss,
                             alpha, num_iteration, epsilon, show_process, save_history)
        self.theta = theta

        return history

    def evaluate(self, X_test: ndarray, y_test: ndarray) -> Tuple[float, float]:
        m = X_test.shape[0]
        X_test = np.hstack([np.ones((m, 1)) * 1.0, X_test])
        loss = _loss(self.theta, X_test, y_test)

        predictions = 1 / (1 + np.exp(-X_test @ self.theta))
        precision = np.count_nonzero((predictions >= 0.5) == y_test) / y_test.size

        return loss, precision

    def predict(self, x: ndarray) -> float:
        """使用模型预测。"""
        x = np.r_[1.0, x]
        if self.theta is not None:
            return _hypothesis(self.theta, x)
        else:
            return None


if __name__ == '__main__':
    pass
