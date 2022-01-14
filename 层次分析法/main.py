import numpy as np
import numpy.linalg as lg
#非一致的判断矩阵(输入矩阵)
A = np.array([[1,2,5],
              [0.5,1,2],
              [0.2,0.5,1]])

RI = np.array([[0,0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54,1.56,1.58,1.59]])

#算术平均法
def fun1(A):
    n = A.shape[0]
    A = A/A.sum(axis=0)
    B = A.sum(axis=1)/n
    print(B)

#几何平均法
def fun2(A):
    n = A.shape[0]
    #转置之后才正确，我也不知道为啥
    A = A.T
    #A = A / A.sum(axis=0)
    B = np.ones(n*1)
    for i in range(n):
        B = B*A[i]
    B = B **(1/n)
    B = B/B.sum()
    print(B)

#一致性检验
def checkout(A):
    n = A.shape[0]
    B = lg.eig(A)
    C = B[0].max()
    CI = (C - n)/(n - 1)
    CR = CI/RI[0][n]
    return CR

#特征值法 * ()
def fun3(A):
    #特征值法之前需一致性检验
    if(checkout(A)<0.1):
        B = lg.eig(A)
        # 最大特征值()
        # B[0]是特征值矩阵，B[0][0]特征值对应的特征矩阵为B[1][:,0]
        C = B[0].max()
        C_index = np.argwhere(B[0] == C)
        D = B[1][:, C_index] / B[1][:, C_index].sum()
        print(D)
    else:
         print("不满足一致性")

fun3(A)