import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as PCHIP
from scipy import interpolate
#拉格朗日插值
def lagrange(x, y):
    M = len(x)
    p = 0.0
    for j in range(M):
        pt = y[j]
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= np.poly1d([1.0, -x[k]])/fac
        p += pt
    print(p)
    return p

'''
Newton插值
'''
#得到插商表
# def get_diff_table(X, Y):
#     n = len(X)
#     A = np.zeros([n, n])
#     for i in range(0, n):
#         A[i][0] = Y[i]
#     for j in range(1, n):
#         for i in range(j, n):
#             A[i][j] = (A[i][j - 1] - A[i - 1][j - 1]) / (X[i] - X[i - j])
#     return A

#计算x点的插值
def newton_interpolation(X,Y,x):
    sum=Y[0]
    temp=np.zeros((len(X),len(X)))
    #将第一行赋值
    for i in range(0,len(X)):
        temp[i,0]=Y[i]
    temp_sum=1.0
    for i in range(1,len(X)):
        #x的多项式
        temp_sum=temp_sum*(x-X[i-1])
        #计算均差
        for j in range(i,len(X)):
            temp[j,i]=(temp[j,i-1]-temp[j-1,i-1])/(X[j]-X[j-i])
        sum+=temp_sum*temp[i,i]
    return sum

#测试牛顿插值法
def newton_test():
    X = [-1, 0, 1, 2, 3, 4, 5]
    Y = [-20, -12, 1, 15, 4, 21, 41]
    xs = np.linspace(np.min(X), np.max(X), 1000, endpoint=True)
    ys = []
    for x in xs:
        ys.append(newton_interpolation(X, Y, x))
    plt.title("newton_interpolation")
    plt.plot(X, Y, 's', label="original values")  # 蓝点表示原来的值
    plt.plot(xs, ys, 'r', label='interpolation values')  # 插值曲线
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.show()


#分段三次埃米尔特插值
def PchipInterpolator():
    x = [1, 2, 3, 4, 5, 6]
    y = [-1, -1, 0, 1, 1, 1]
    pchip = PCHIP(x, y)
    x_new = [2.5, 3.5, 4.5]
    y_new = pchip(x_new)
    plt.plot(x, y, 's', label="original values")  # 蓝点表示原来的值
    plt.plot(x, y, 'b-')  # 原始点为蓝色线条
    plt.plot(x_new, y_new, 'ro')  # 插值点为红色圆点
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.show()

# 三次样条插值
def spline():
    x = [-1, 0, 1, 2, 3, 4, 5]
    y = [-20, -12, 1, 15, 4, 21, 41]
    xs = np.linspace(np.min(x), np.max(x), 1000, endpoint=True)
    tck = interpolate.splrep(x, y)
    ys = interpolate.splev(xs, tck, der=0)
    plt.title("spline")
    plt.plot(x, y, 's', label="original values")  # 蓝点表示原来的值
    plt.plot(xs, ys, 'r', label='interpolation values')  # 插值曲线
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.show()