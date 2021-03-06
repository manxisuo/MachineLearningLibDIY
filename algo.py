# encoding: utf-8
from typing import Tuple
from time import time
import numpy as np
from numpy import ndarray
from tool import CachedFunc


class History:
    """表示模型训练过程的历史记录"""
    def __init__(self, loss_list, consuming_time):
        self.loss_list = loss_list  # 每次迭代后的损失函数的值的列表
        self.consuming_time = consuming_time  # 训练耗时


def bgd(X: ndarray, y: ndarray, _loss, _gradient_of_loss,
        alpha=0.01, num_iteration: int = None, epsilon: float = None,
        show_process=False, save_history=False) -> Tuple[ndarray, History]:
    """批量梯度下降。"""
    start = time()
    theta = np.zeros(X.shape[1])  # 模型参数初始化
    loss_list = []  # 每次迭代后的损失的列表
    k = 0  # 迭代次数
    previous_loss = _loss(theta, X, y)  # 前一次的损失

    func_loss = CachedFunc(_loss)  # 为了减少loss的计算次数

    while True:
        theta = theta - _gradient_of_loss(theta, X, y) * alpha

        # 检查迭代次数条件是否满足
        k += 1
        if num_iteration and 0 < num_iteration < k:
            break

        # 检查损失下降条件是否满足
        if epsilon and 0 < epsilon:
            if abs(func_loss(theta, X, y) - previous_loss) <= epsilon:
                break
            previous_loss = func_loss(theta, X, y)

        if show_process:
            print(f'epoch: {k}, loss: {func_loss(theta, X, y)}, theta: {theta}')

        if save_history:
            loss_list.append(func_loss(theta, X, y))

        func_loss.reset()

    return theta, History(loss_list, time() - start)
