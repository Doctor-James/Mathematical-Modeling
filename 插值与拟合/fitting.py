import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

def Fun(p, x):  # 定义拟合函数形式
    a1, a2, a3 = p
    #return a1 * x ** 2 + a2 * x + a3
    return a1 * np.exp(x) + a3

def error(p, x, y):  # 拟合残差
    return Fun(p, x) - y

#最小二乘法
def fitting_1():
    x = np.linspace(0, 10, 100)  # 创建时间序列
    p_value = [-2, 5, 10]  # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    y = Fun(p_value, x) + noise * 2  # 加上噪声的序列


    p0 = [0.1, -0.01, 100]  # 拟合的初始参数设置
    para = leastsq(error, p0, args=(x, y))  # 进行拟合
    y_fitted = Fun(para[0], x)  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend(loc=4)
    plt.show()


#多项式拟合
def fitting_2():
    #生成数据
    x = np.linspace(-10, 10, 100)  # 创建时间序列
    p_value = [-2, 5, 10]  # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    #y = Fun(p_value, x) + noise * 2  # 加上噪声的序列
    y = np.cos(x) # 加上噪声的序列
    z1 = np.polyfit(x, y, 10)  # 用2次多项式拟合，可改变多项式阶数；
    p1 = np.poly1d(z1)  # 得到多项式系数，按照阶数从高到低排列

    y_fitted = np.polyval(z1, x)

    plt.figure
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend(loc=4)
    plt.show()


def BP_fitting():
    x = np.linspace(-3, 3, 600)
    # print(x)
    # print(x[1])
    x_size = x.size
    y = np.zeros((x_size, 1))
    # print(y.size)
    for i in range(x_size):
        y[i] = math.sin(x[i])

    # print(y)

    hidesize = 10
    W1 = np.random.random((hidesize, 1))  # 输入层与隐层之间的权重
    B1 = np.random.random((hidesize, 1))  # 隐含层神经元的阈值
    W2 = np.random.random((1, hidesize))  # 隐含层与输出层之间的权重
    B2 = np.random.random((1, 1))  # 输出层神经元的阈值
    threshold = 0.005
    max_steps = 100

    def sigmoid(x_):
        y_ = 1 / (1 + math.exp(-x_))
        return y_

    E = np.zeros((max_steps, 1))  # 误差随迭代次数的变化
    Y = np.zeros((x_size, 1))  # 模型的输出结果
    for k in range(max_steps):
        temp = 0
        for i in range(x_size):
            hide_in = np.dot(x[i], W1) - B1  # 隐含层输入数据
            # print(x[i])
            hide_out = np.zeros((hidesize, 1))  # 隐含层的输出数据
            for j in range(hidesize):
                # print("第{}个的值是{}".format(j,hide_in[j]))
                # print(j,sigmoid(j))
                hide_out[j] = sigmoid(hide_in[j])
                # print("第{}个的值是{}".format(j, hide_out[j]))

            # print(hide_out[3])
            y_out = np.dot(W2, hide_out) - B2  # 模型输出
            # print(y_out)

            Y[i] = y_out
            # print(i,Y[i])

            e = y_out - y[i]  # 模型输出减去实际结果。得出误差

            ##反馈，修改参数
            dB2 = -1 * threshold * e
            dW2 = e * threshold * np.transpose(hide_out)
            dB1 = np.zeros((hidesize, 1))
            for j in range(hidesize):
                dB1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * (-1) * e * threshold)

            dW1 = np.zeros((hidesize, 1))

            for j in range(hidesize):
                dW1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * x[i] * e * threshold)

            W1 = W1 - dW1
            B1 = B1 - dB1
            W2 = W2 - dW2
            B2 = B2 - dB2
            temp = temp + abs(e)

        E[k] = temp

        if k % 100 == 0:
            print(k)

    plt.figure()
    plt.plot(x, y,label='Original curve')
    plt.plot(x, Y, color='red', linestyle='--',label='Fitted curve')
    plt.legend(loc=4)
    plt.show()

BP_fitting()
