import numpy as np
from matplotlib import pyplot as plt
from func import example, chazhi, K, p_target, connect
from sklearn.metrics import mean_squared_error
import torch
# 维数和样本数目不同
samp_num = np.array([220000, 80000])  # 2个level对应的样本数目

L = 2  # mlmcmc的层数
np.random.seed(42)
dimension = np.array([41, 81])
samp = [np.zeros((41, 220000)), np.zeros((81, 80000))] # 需要手动改动
samp_per_layer = [np.zeros((41, 220000)), np.zeros((81, 80000))]
beta = 0.001
beta0 = 0.004
beta1 = 0.004
plt.close('all')


# 真实数据,分成3个维度便于后续的讨论.
data = example(dimension[L-1])

data0 = []
for k in range(dimension[L-1]):
    if k % 2 == 0:
        data0.append(data[k])
data0 = np.array(data0)



DATA = []
# DATA.append(data1)
DATA.append(data0)
DATA.append(data)



K_all = []
K_inv = []
Sigma = 0.02**2
for l in range(L):
    K_l = K(dimension[l])
    Kl_inv = np.linalg.inv(K_l+1e-8*np.eye(dimension[l]))
    # Kl_inv = np.linalg.inv(K_l)
    K_all.append(K_l)
    K_inv.append(Kl_inv)


# B代表的是每一个维度的采样结果
B = []
SAMPLE = []
burn_in = [40000, 20000]
# burn_in = [0, 0]
for l in range(L):
    accept_num = 0
    accept_num1 = 0
    accept_num2 = 0
    A = np.zeros((dimension[l], 1))
    sample_mean_perlayer1= np.zeros((dimension[0], 1))
    sample_mean_perlayer2 = np.zeros((dimension[1], 1))
    arr = np.ones((dimension[l], 1)) * 0.1
    mean = np.zeros(dimension[l])
    cov = K_all[l]

    # 用于当l不等于0
    if l != 0:
        # on level l-1
        arr_0 = np.ones((dimension[l - 1], 1)) * 0.1
        # arr_0 = DATA[0]
        mean_0 = np.zeros(dimension[l - 1])
        cov_0 = K_all[l - 1]

        # level l
        arr_c = np.ones((dimension[l - 1], 1)) * 0.1
        # arr_c = DATA[0]
        # fine部分的初始化
        delta_dimension = dimension[l] - dimension[l - 1]
        arr_f = np.ones((delta_dimension, 1)) * 0.1
        mean_f = np.zeros(delta_dimension)
        #
        arr = connect(arr_c, arr_f)

        delta_t = 1 / delta_dimension
        K_1 = K(delta_dimension)
        cov_f = K_1
        K_1_inv = np.linalg.inv(K_1+1e-8*np.eye(delta_dimension))
        data_arr = example(delta_dimension)

#  接下来是mlmcmc的部分
    for i in range(samp_num[0]):
        if l == 0:
            accept_rate1, accept_rate2 = 0,0
            pass
            # # Z = np.random.randn(dimension, 1)
            # # arr_star = (1 - beta ** 2) * arr + beta * Z
            # Z = np.random.multivariate_normal(mean, cov, (1,))
            # arr_star = (1 - beta**2)**0.5*arr + beta * Z.T
            # # arr是x，arr_star是x*，作为下一步的候选值。都是d*1矩阵
            # # px_part = p_target(data0, arr_star, dimension[l], K_inv[l]) - p_target(data0, arr, dimension[l], K_inv[l])
            # px_part = p_target(DATA[l], arr_star, dimension[l])[l] - p_target(DATA[l], arr, dimension[l])[l]
            # # qx_part = q(arr, arr_star)-q(arr_star, arr)
            # alpha1 = min(1, np.exp(px_part))
            # # print(alpha1)
            # m = np.random.rand(1)[0]
            # if m < alpha1:
            #     arr = arr_star
            #     accept_num += 1
            # samp[l][:, [i]] = arr
            # samp_per_layer[l][:, [i]] = arr
            # accept_rate = accept_num / samp_num[l]
            # accept_rate1 = 0
            # accept_rate2 = 0
        else:
            # for i in range(samp_num[0]):
            # on level l-1
            Z = np.random.multivariate_normal(mean_0, cov_0, (1,))
            arr_star_0 = (1 - beta0 ** 2) ** 0.5 * arr_0 + beta0 * Z.T
            # px_part0 = p_target(data0, arr_star_0, dimension[l-1], K_inv[l-1]) - p_target(data0, arr_0, dimension[l-1], K_inv[l-1])
            px_part0 = p_target(DATA[l-1], arr_star_0)[l-1] - p_target(DATA[l-1], arr_0)[l-1]
            alpha0 = min(1, np.exp(px_part0))
            m = np.random.rand(1)[0]
            if m < alpha0:
                arr_0 = arr_star_0
                accept_num1 += 1
            accept_rate1 = accept_num1 / samp_num[0]
            # print("l=%f" % l, accept_rate)
            samp_per_layer[0][:, [i]] = arr_0

            if i < samp_num[1]:
                # on level l
                # 首先使用pcn生成FINE部分，这个部分的维数是delta_dimension
                Z_f = np.random.multivariate_normal(mean_f, cov_f, (1,))
                arr_star_f = (1 - beta1 ** 2) ** 0.5 * arr_f + beta1 * Z_f.T
                # 将粗糙部分与F部分连接在一起
                arr_star_c = arr_0
                arr_star = connect(arr_star_c, arr_star_f)

                p_part_c = p_target(DATA[l-1], arr_c)[l-1] - p_target(DATA[l-1], arr_star_c)[l-1]
                p_part = p_target(DATA[l], arr_star)[l] - p_target(DATA[l], arr)[l]
                alpha1 = min(1, np.exp(p_part + p_part_c))
                m = np.random.rand(1)[0]
                if m < alpha1:
                    arr = arr_star
                    arr_c = arr_star_c
                    arr_f = arr_star_f
                    accept_num2 += 1
                accept_rate2 = accept_num2 / samp_num[1]
                samp_per_layer[1][:, [i]] = arr
            arr0_new = chazhi(arr_0)
            # samp[l][:, [i]] = arr - arr0_new
            # print(accept_rate1)
    print(accept_rate1,accept_rate2)

samp_per_layer[0] = samp_per_layer[0][:, burn_in[0]:]
samp_per_layer[1] = samp_per_layer[1][:, burn_in[1]:]
for k in range(dimension[0]):
    sample_mean_perlayer1[k] = np.mean(samp_per_layer[0][k])
SAMPLE.append(sample_mean_perlayer1)
for k in range(dimension[1]):
    sample_mean_perlayer2[k] = np.mean(samp_per_layer[1][k])
SAMPLE.append(sample_mean_perlayer2)

SAMPLE0_chazhi = chazhi(SAMPLE[0])


x = np.linspace(0, 1, 81)
interval0 = [1 if (i < 1//3) else 0 for i in x]
interval1 = [1 if (1/3 <= i < 2/3) else 0 for i in x]
interval2 = [1 if (i >= 2/3) else 0 for i in x]
u = 0 * interval0 + 1 * interval1 + 0*interval2

# ########################################################
# plt.figure(figsize=(10, 5))
# plt.plot(x, u, 'k', lw=3, label='x')
# plt.plot(x, data, '.k', label='y=x+n')
# plt.plot(x, data_mlmcmc, 'k', color='r', label='x*')  # 可能有问题
#
# plt.legend()
# # plt.title('Model and data')
# plt.title('Model and data,beta0 =%f,beta1 =%f'%(beta0,beta1))
# plt.show()
plt.figure(figsize=(10, 5))
plt.plot(x, u, 'k', lw=3, label='x')
# plt.plot(x, data_mlmcmc, 'k', color='y', label='x*')  # 可能有问题
plt.plot(x,SAMPLE0_chazhi,color='b',label='1')
plt.plot(x,SAMPLE[1],color='c',label='2')
# plt.plot(x, B[1], label = 'chazhi')
# plt.plot(x,SAMPLE[2],color='g',label='3')
plt.legend()
plt.title('Model and data,beta0 =%f,beta1 =%f'%(beta0,beta1))
plt.show()
