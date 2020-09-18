# encoding: utf-8
import csv
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionModel
from util import Normalizer


def load_boston_data():
    """
    加载波士顿房价数据集。

    共506个样本，13个特征。
    """
    with open('data/boston.csv', 'r') as f:
        reader = csv.reader(f)
        *xlabels, ylabel = next(reader)
        xs, ys = [], []
        for line in reader:
            *x, y = line
            xs.append([float(x_i) for x_i in x])
            ys.append(float(y))
        return xs, ys, xlabels, ylabel


if __name__ == '__main__':
    # 加载数据，并将数据归一化
    xs, ys, xlabels, ylabel = load_boston_data()
    normalizer = Normalizer()
    normalizer.fit(xs)
    xs = normalizer.transform(xs)

    # 划分训练集、测试集
    xs_training, ys_training = xs[:400], ys[:400]
    xs_test, ys_test = xs[400:], ys[400:]

    # 训练模型
    model = LinearRegressionModel()
    history = model.fit(xs_training, ys_training,
                        alpha=0.001, num_iteration=300, epsilon=1e-7,
                        show_process=False, save_history=True)

    # 训练结果
    print('参数θ:', model.theta)
    print('训练集损失:', history.loss_list[-1])
    print('耗时：', history.consuming_time)

    # 画出训练过程中的loss变化曲线
    plt.plot(range(1, len(history.loss_list) + 1), history.loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # 评估模型
    print('测试集损失:', model.evaluate(xs_test, ys_test))
