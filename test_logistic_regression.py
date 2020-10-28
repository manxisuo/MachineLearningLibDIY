# encoding: utf-8
import csv
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionModel
from util import Normalizer
import numpy as np


def converter(bs):
    s = str(bs, encoding='ascii')
    return 1.0 if s == 'setosa' else 0.0  # 将setosa类作为正例（即y=1）


def load_iris_data():
    data = np.genfromtxt('data/iris.csv',delimiter=',', skip_header=True, converters={4: converter})
    np.random.shuffle(data)  # 由于数据集是有序的，因此需要打乱
    X, y = data[:, :-1], data[:, -1:]
    y = y.reshape(y.size)
    return X, y


if __name__ == '__main__':
    X, y = load_iris_data()

    normalizer = Normalizer()
    normalizer.fit(X)
    X = normalizer.transform(X)

    # 划分训练集、测试集
    X_training, y_training = X[:120, :], y[:120]
    X_test, y_test = X[120:, :], y[120:]

    # 训练模型
    model = LogisticRegressionModel()
    history = model.fit(X_training, y_training,
                        alpha=0.03, num_iteration=50000, epsilon=None,
                        show_process=False, save_history=True)

    # 训练结果
    print('耗时：', history.consuming_time)
    print('参数θ:', model.theta)
    print('迭代数:', len(history.loss_list))
    print('训练集损失:', history.loss_list[-1])

    # 画出训练过程中的loss变化曲线
    plt.plot(range(1, len(history.loss_list) + 1), history.loss_list)
    plt.title('predict setosa')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # 评估模型
    loss, precision = model.evaluate(X_test, y_test)
    print('测试集损失:', loss)
    print('测试集准确率:', precision)