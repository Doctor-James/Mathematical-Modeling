# 插值

## 一 . 拉格朗日插值

![image-20220115191051354](/home/zjl/.config/Typora/typora-user-images/image-20220115191051354.png)

**数据大面积缺失时，拉格朗日的插值结果很差。**

## 二 . 牛顿法插值

## 三 . 分段线性插值

如拉格朗日插值，牛顿法插值等高次插值，n越大，越容易震荡，遂用低次多项式插值

![image-20220115194806035](/home/zjl/.config/Typora/typora-user-images/image-20220115194806035.png)

即用多个小直线段来连接

局限性：新的x只能介于样本点的x区间内

### 分段三次埃米尔特插值法（*）

较为常用，只准备此方法的代码块

![image-20220115195151744](/home/zjl/.config/Typora/typora-user-images/image-20220115195151744.png)

![image-20220115195204133](/home/zjl/.config/Typora/typora-user-images/image-20220115195204133.png)

## 四 . 三次样条插值

![image-20220115201619927](/home/zjl/.config/Typora/typora-user-images/image-20220115201619927.png)

适用于对插值函数光滑性（多阶导数连续）要求较高的场景



## 五 . 二维插值

上述均为一维插值



​		**网格节点**插值适用于节点比较规范的情况，即在包含所给节点的矩形区域内，节点由两族平行于坐标轴的直线的交点所组成。主要方法有：最邻近插值（选取待插值点最邻近已知数据点的函数值作为待插值节点的值）、分片线性插值和双线性插值。
  **散乱节点**插值适用于一般的节点，多用于节点不太规范（即节点为两族平行于坐标轴的直线的部分交点）的情况

### 1 . 网格节点插值法

![image-20220115210715116](/home/zjl/.config/Typora/typora-user-images/image-20220115210715116.png)

直接调用api，本质上还是三次样条插值

### 2 . 散乱节点插值法



# 拟合

## 一 . 最小二乘法

可以用来做非线性拟合

## 二 . 多项式拟合



# 函数逼近

## 一 . 多项式拟合

可以用来做函数逼近，比如用一个十次多项式逼近cos

![image-20220115223501199](/home/zjl/.config/Typora/typora-user-images/image-20220115223501199.png)

## 二 . bp神经网络

迭代100次

![image-20220115223946624](/home/zjl/.config/Typora/typora-user-images/image-20220115223946624.png)
