import numpy as np
from scipy import optimize as op


def solve_line():
    # 给出变量取值范围
    x1 = (0, None)
    x2 = (0, None)
    x3 = (0, None)

    # 目标函数系数,3x1列向量
    c = np.array([-2, -3, 5])
    # 不等式约束系数A，2x3维矩阵
    A_ub = np.array([[-2, 5, -1], [1, 3, 1]])
    # 不等式约束B，2x1维矩阵
    B_ub = np.array([-10, 12])
    # 等式约束系数Aeq，3x1维列向量
    A_eq = np.array([[1, 1, 1]])
    # 等式约束beq，1x1数值
    B_eq = np.array([7])

    res = op.linprog(c, A_ub, B_ub, A_eq, B_eq, bounds=(x1, x2, x3))  # 调用函数进行求解
    print(res)
