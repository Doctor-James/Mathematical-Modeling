import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import matplotlib.cm as cm

def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2 + y**2))
'''
若为散乱数据，唯一的区别是将
newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')
改为
newfunc = interpolate.griddata(x, y, fvals, kind='cubic')
'''
#二维插值，节点为网格数据
def interp2d_1():
    # X-Y轴分为15*15的网格
    #二维的表示是np.mgrid[起点：终点：步长，起点：终点：步长],15j表示把从起点到终点等分成15个点(左闭右闭等分)
    y, x = np.mgrid[-1:1:15j, -1:1:15j]
    fvals = func(x,y) # 计算每个网格点上的函数值  15*15的值
    # 三次样条二维插值
    newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')
    # 计算100*100的网格上的插值
    xnew = np.linspace(-1, 1, 100)  # x
    ynew = np.linspace(-1, 1, 100)  # y
    fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值

    # 绘图
    # 为了更明显地比较插值前后的区别，使用关键字参数interpolation='nearest'
    # 关闭imshow()内置的插值运算。
    #plt.subplot(2,1,1)
    im1 = plt.imshow(fvals, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='none', origin="lower")  # plt.cm.jet
    # extent=[-1,1,-1,1]为x,y范围  favals为
    plt.colorbar(im1)

    #plt.subpltot(2,1,2)
    im2 = plt.imshow(fnew, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
    plt.colorbar(im2)
    plt.show()

#二维插值，节点为网格数据,三维展示方法
def interp2d_2():
    # X-Y轴分为20*20的网格
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    x, y = np.meshgrid(x, y)  # 20*20的网格数据

    fvals = func(x, y)  # 计算每个网格点上的函数值  15*15的值

    fig = plt.figure(figsize=(9, 6))
    # Draw sub-graph1
    ax = plt.subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(x, y, fvals, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.colorbar(surf, shrink=0.5, aspect=5)  # 标注

    # 二维插值
    newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')  # newfunc为一个函数

    # 计算100*100的网格上的插值
    xnew = np.linspace(-1, 1, 100)  # x
    ynew = np.linspace(-1, 1, 100)  # y
    fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值  np.shape(fnew) is 100*100
    xnew, ynew = np.meshgrid(xnew, ynew)
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
    ax2.set_xlabel('xnew')
    ax2.set_ylabel('ynew')
    ax2.set_zlabel('fnew(x, y)')
    plt.colorbar(surf2, shrink=0.5, aspect=5)  # 标注

    plt.show()


interp2d_2()