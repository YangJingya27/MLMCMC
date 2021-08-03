import numpy as np


# 定义不同维数时的真实数据
def example(dimension):
    x = np.linspace(0, 1, dimension)
    interval0 = [1 if (i < 1 // 3) else 0 for i in x]
    interval1 = [1 if (1 / 3 <= i < 2 / 3) else 0 for i in x]
    interval2 = [1 if (i >= 2 / 3) else 0 for i in x]
    u = 0 * interval0 + 1 * interval1 + 0 * interval2
    n = np.random.normal(0, 0.02, dimension)
    y = u + n
    y = y.reshape((dimension,1))
    return y


# 混合先验中的TV项
def DX_1(xy, dim):
    DX1 = 0
    for z in range(dim - 1):  # 这里的维度是变化的
        dx_1 = abs(xy[z + 1] - xy[z])
        DX1 = DX1 + dx_1
    return DX1


############################################################针对三层的情况
lam = 300
Sigma2 = 0.02**2
Sigma1 = (1+1/80) * Sigma2
Sigma0 = (1+1/40) * Sigma1


# 混合先验 target distribution 的指数部分
def p_target(y_arr, u_arr, dim):
    # part_u = (u_arr.T.dot(k)).dot(u_arr)
    part_y0 = (y_arr-u_arr).T.dot(y_arr-u_arr) * 1/Sigma0
    part_y1 = (y_arr - u_arr).T.dot(y_arr - u_arr) * 1 / Sigma1
    part_y2 = (y_arr - u_arr).T.dot(y_arr - u_arr) * 1 / Sigma2
    # X = 0.5 * (part_u + part_y)+lam * DX_1(u_arr, dim)
    X0 = 0.5 * part_y0 + lam * DX_1(u_arr, dim)
    X1 = 0.5 * part_y1 + lam * DX_1(u_arr, dim)
    X2 = 0.5 * part_y2 + lam * DX_1(u_arr, dim)
    # 每一个X代表第n层的取值，由于每一层的似然函数的方差不同，所以分别表示
    return -1*X0[0, 0], -1*X1[0, 0], -1*X2[0, 0]


def chazhi(arr):
    arr_new = np.zeros((2*arr.size-1, 1), dtype=object)
    for t in range(arr_new.size):
        t = int(t)
        if t % 2 == 0:
            arr_new[t] = arr[t // 2]
        else:
            arr_new[t] = (arr[(t + 1) // 2] + arr[(t - 1) // 2]) * 0.5
    return arr_new


d = 0.02
r = 0.1


# 先验分布中的协方差函数
def K(dimension):
    delta_t = 1/dimension
    # 定义协方差矩阵K
    K_l = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            K_l[i, j] = r * np.exp(-0.5 * (abs(i-j) * delta_t / d)**2)
    return K_l


def connect(c_part,f_part):
    connection = np.zeros((c_part.size+f_part.size,1))
    for i in range(c_part.size + f_part.size):
        if i % 2 == 0:
            connection[i] = c_part[i // 2]
        else:
            connection[i] = f_part[(i - 1) // 2]
    return connection