"""
这个脚本是用于验证Instance Normalization的计算过程，辅助学习Instance Normalization的原理的。
"""
import torch
from torch import nn
import numpy as np

data = torch.rand((2, 2, 2, 2), dtype=torch.float32)  # (4,3,10,12)
norm = nn.InstanceNorm2d(3)
# data_mean = torch.mean(data, dim=[2, 3])
# data_mean = data.view((2,2,-1))
data_mean = data.view((2 * 2, -1))
data_mean = torch.mean(data_mean, 1)
data_mean = data_mean.view((2, 2, 1, 1))
print('data_mean shape', data_mean.shape)
# data_mean = data_mean[:, :, None, None]
# data_std = torch.std(data, dim=[2, 3])
data_std = data.view((2 * 2, -1))
data_std = torch.std(data_std, 1, unbiased=False)
data_std = data_std.view((2, 2, 1, 1))
# data_std = data_std[:, :, None, None]
data_simulate = (data - data_mean) / data_std
result = norm(data)
# print('result of norm == result of simulation', result - data_simulate)
print(torch.max(torch.abs(result - data_simulate) / torch.abs(data)))
# 不设置unbiased=False,输出了21.0818，也就是说有的相对误差是20多倍，这就说明上面的计算是不等价的。
# 对比下面别人写的，也设置了unbiased=False后，就得到了0.0013，肉眼观察后也几乎是相等的。然后看官方文档后搞懂了，unbiased是无偏估计
print(result)
print('=' * 20)
print(data_simulate)
# https://www.csdn.net/tags/NtTacg4sNDgzNDktYmxvZwO0O0OO0O0O.html
# import torch
# import torch.nn as nn
#
# # from torch_geometric.data import DataLoader
# def D1(p,z):
#     norm = nn.InstanceNorm1d(1)
#     p = norm(p)
#     print(f'D1 norm P: {p}')
#     # z = torch.instance_norm(z,dim=1)
#     z = norm(z)
#     print(f'D1 z: {z}')
#     return -(p*z).sum(dim=1).mean()
#
# def D2(p,z):
#     # norm = nn.InstanceNorm1d(1)
#     # manually mormalization
#     N,C = p.size(0),p.size(1)
#     p_t = p.reshape(N * C, -1)
#     p_mean = p_t.mean(dim=1).reshape(N,C,-1)
#     p_std = p_t.std(dim=1,unbiased=False).reshape(N,C,-1)
#     p = (p-p_mean)/p_std
#     print(f'p_mean: {p_mean},\n p_std: {p_std},\n D2 p_norm: {p}')
#
#     z_t = z.reshape(N * C, -1)
#     z_mean = z_t.mean(dim=1).reshape(N, C, -1)
#     z_std = z_t.std(dim=1, unbiased=False).reshape(N, C, -1)
#     z = (z - z_mean) / z_std
#     print(f'z_mean: {z_mean}, \n z_std: {z_std},\n D2 z_norm: {z}')
#     return -(p*z).sum(dim=1).mean()
#
# if __name__ == '__main__':
#     # similarity function D(p,z)
#     torch.manual_seed(3)
#
#     p = torch.randn(2,5)
#     z = torch.randn(2, 5)
#     print(f'p:{p}, \n z:{z}')
#
#     N,C = p.size(0),p.size(1)
#     p = p.reshape(N, 1, -1)
#
#     z = z.reshape(N,1,-1)
#
#     print(D1(p,z))
#     print(D2(p,z))
