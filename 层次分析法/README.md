# 层次分析法

![image-20220114162450893](/home/zjl/.config/Typora/typora-user-images/image-20220114162450893.png)

## 两两比较，构造判断矩阵

![image-20220114162818349](/home/zjl/.config/Typora/typora-user-images/image-20220114162818349.png)

![image-20220114162937687](/home/zjl/.config/Typora/typora-user-images/image-20220114162937687.png)

## 一致矩阵

![image-20220114163523236](/home/zjl/.config/Typora/typora-user-images/image-20220114163523236.png)

### 一致性检验

![image-20220114163706887](/home/zjl/.config/Typora/typora-user-images/image-20220114163706887.png)

![image-20220114164126153](/home/zjl/.config/Typora/typora-user-images/image-20220114164126153.png)

其中：lamda~max~为判断矩阵最大的特征值，n为判断矩阵的维数（当lamda~max~ == n时，CI=0，完全一致）

## 通过判断矩阵求权重

### 1. 判断矩阵是一致矩阵

![image-20220114165914433](/home/zjl/.config/Typora/typora-user-images/image-20220114165914433.png)

### 2. 判断矩阵是非一致矩阵

#### 2.1 算术平均法

![image-20220114170100070](/home/zjl/.config/Typora/typora-user-images/image-20220114170100070.png)

每一列分别归一化，直接相加除以3

#### 2.2 几何平均法

![image-20220114170201667](/home/zjl/.config/Typora/typora-user-images/image-20220114170201667.png)

#### 2.3 特征值法

![image-20220114170343724](/home/zjl/.config/Typora/typora-user-images/image-20220114170343724.png)

## 总结

用途：将某些定性的量做定量化处理

步骤：

1. 两两比较，求出判断矩阵
2. 检验判断矩阵一致性，若满足CR<0.1,可求权重；若不满足，需要修改判断矩阵
3. 用三种方法求权重（最好用特征值法）
