# encoding: utf-8
import csv
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionModel
from util import Normalizer
import numpy as np


def load_boston_data():
    """加载波士顿房价数据集。共506个样本，13个特征。"""
    data = np.genfromtxt('data/boston.csv', delimiter=',', skip_header=True)
    X, y = data[:, :-1], data[:, -1:]
    y = y.reshape(y.size)
    return X, y


if __name__ == '__main__':
    # 加载数据，并将数据归一化
    X, y = load_boston_data()
    normalizer = Normalizer()
    normalizer.fit(X)
    X = normalizer.transform(X)

    # 划分训练集、测试集
    X_training, y_training = X[:400, :], y[:400]
    X_test, y_test = X[400:, :], y[400:]

    # 训练模型
    model = LinearRegressionModel()
    history = model.fit(X_training, y_training,
                        alpha=0.003, num_iteration=50000, epsilon=None,
                        show_process=False, save_history=True)

    # 训练结果
    print('耗时：', history.consuming_time)
    print('参数θ:', model.theta)
    print('迭代数:', len(history.loss_list))
    print('训练集损失:', history.loss_list[-1])

    # 画出训练过程中的loss变化曲线
    plt.plot(range(1, len(history.loss_list) + 1), history.loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('predict house price')
    plt.show()

    # 评估模型
    loss = model.evaluate(X_test, y_test)
    print('测试集损失:', loss)
