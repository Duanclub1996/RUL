import os
import sys
import math
import tqdm
import time
import glob
import scipy
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from scipy import stats
from pathlib import Path
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange, repeat
from sklearn.preprocessing import MinMaxScaler
from einops.layers.torch import Rearrange, Reduce
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

file_path = './35Hz12kN/Bearing1_1/'
file_name = glob.glob(file_path+'*.csv')


def load_data(file_path, file_name):
    all_data = []
    for i in range(len(file_name)):
        data = pd.read_csv(file_path + f'{i+1}.csv').values
        all_data.append(data)
    return np.stack(all_data, axis=0).reshape(-1, 2)[:, 0]  # (4030464, 1)
# data = load_data(file_path=file_path,file_name=file_name)


# 读取数据
def data_loader(raw_data, win_size, step):
    use_data = []
    # data = raw_data.values  # 转换为numpy
    data_num = (raw_data.shape[0] - win_size) // step - 1  # 计算样本数量
    print(f'样本数量为{data_num}')
    for i in range(data_num):
        start = i * step
        _data = raw_data[start:start + win_size]
        use_data.append(_data)
    return np.array(np.stack(use_data, axis=0))

# 获得时域和频域特征


def get_features(y):
    # 最大值
    max_y = np.max(y)
    # 最小值
    min_y = np.min(y)
    # 中值
    median_y = np.median(y)
    # 平均值
    mean_y = np.mean(y)
    # 方差
    var_y = np.var(y)
    # 均方根
    rms = np.sqrt((np.mean(y**2)))
    # 峰值
    peak = max(abs(max_y), abs(min_y))
    # 峰峰值
    peak2peak = max_y - min_y
    # 峰值因子
    crestf = max_y / rms if rms != 0 else 0
    xr = np.square(np.mean(np.sqrt(np.abs(y))))
    # 裕度
    margin = (max_y / xr) if xr != 0 else 0
    yr = np.mean(np.abs(y))
    # 脉冲因子
    pulse = max_y / yr if yr != 0 else 0
    # 波形因子
    waveform = rms / yr if yr != 0 else 0
    # 峭度
    kur = scipy.stats.kurtosis(y)
    # 偏斜度
    sk = scipy.stats.skew(y)
    return max_y, min_y, median_y, mean_y, var_y, peak, peak2peak, rms, crestf, margin, pulse, waveform, kur, sk

# 根据窗口提取特征


def feature_ex(data):
    feature_list = []
    for i in range(data.shape[0]):
        _feature = get_features(data[i, :])
        feature_list.append(_feature)
    feature = np.stack(feature_list, axis=0)
    return feature


# 获得所有特征
def get_time_f_feature(filepath, filename, win_size, step):
    data = load_data(file_path=filepath, file_name=filename)
    win_data = data_loader(data, win_size, step)
    feature = feature_ex(win_data)
    return feature

#


feature_name = ['最大值', '最小值', '中值', '平均值', '方差', '均方根',
                '峰值', '峰峰值', '峰值因子', '裕度', '脉冲因子', '波形因子', '峭度', '偏斜度']

# KS检验


def KS_test(data):
    ks_score = []
    for i in range(data.shape[1]):
        s = data[:, i]
        u = np.mean(s)  # 计算均值
        std = np.std(s)  # 计算标准差
        p = stats.kstest(s, 'norm', (u, std))
        ks_score.append(p[1])
    return ks_score

# print(KS_test(allfeature))


# KS检验结果
# [2.4e-181, 2.3e-153, 1.9e-63, 0.48700770792524095, 0.0, 5.1e-163, 1.00e-176, 1.3e-306, 6.93e-31, 3.4e-19, 1.9e-24, 1.9e-29, 1.7e-87, 1.4e-07]

# 获取label   3/sigma 原则
# def get_label(x):
#     # 获取label
#     label = []
#     final_len = len(x) - 1
#     count = 0
#     thr = np.mean(x) + 3*np.std(x)
#     for i in range(x.shape[0]):
#        if  x[i] > thr :
#            count = count + 1
#        if count >= 2:
#            FPT = i
#            print(i+1)
#            break
#     for j in range(len(x)):
#         if j <= FPT:
#             label.append(1)
#         else:

#             label.append(1 - ((j - FPT)/ (final_len - FPT)))

#     return np.array(label)


def get_IF_label(IF, data):
    """
    构建 RUL 标签

    IF：退化点
    data: 数据集
    """
    label = []

    end = data.shape[0] - 1

    for i in range(len(data)):
        if i <= IF:
            label.append(1)
        else:
            rul = 1 - (i - IF) / (end - IF)
            label.append(rul)

    return label


# 可视化
# all_feature 一般输入所有的特征  label：标签  save_dir： 保存地址 feature_name：特征名称
def vis(all_feature, label, save_dir, feature_name):
    fig1, ax = plt.subplots(14, 1, figsize=(10, 60))
    for i in range(14):
        # sns.lineplot(data=all_feature[:,i],label=feature_name[i],ax=ax[i])
        sns.lineplot(data=all_feature[:, i], ax=ax[i])
        ax[i].set_title(feature_name[i])
        # ax[i].legend(loc='upper right')
    fig1.savefig(f'{save_dir}' + 'all_feature.pdf', dpi=800)
    plt.close(fig1)

    fig2 = plt.figure()
    sns.lineplot(data=label, label='label')
    plt.title("RUL(FPT-Point:2583)")
    fig2.savefig(f'{save_dir}' + 'label.pdf', dpi=800)
    plt.close(fig2)





# allfeature = get_time_f_feature(filepath=file_path,filename=file_name,win_size=1024,step=1024)

# vis(all_feature=allfeature,label=get_label(allfeature[:,3]),save_dir='./',feature_name=feature_name)


# 退化点来源：L. Xiao, Z. Liu, Y. Zhang, Y. Zheng, C. Cheng, Degradation assessment of bearings with trend-reconstruct-based
# features selection and gated recurrent unit network, Measurement, 2020, Vol 165, Art no. 108064.
# get_label
def get_IF(HI):
    '''
      HI: 设置的健康曲线
    '''
    for i in range(len(HI)):
        mean = np.mean(HI[0:i])
        std = np.std(HI[0:i])
        thr = mean + 3 * std
        HI_flge = 0
        stop_flage = 2
        if HI[i] > thr:
            HI_flge += 1
            if HI_flge > stop_flage:
                IF = i
                break
        label = get_IF_label(IF, HI)
        return label




class dataset(Dataset):
    def __init__(self, file_path, file_name, win_size=1024, step=1024,enc_step = 20,enc_len=20,pre_len=1):
        super().__init__()
        self.enc_len = enc_len
        self.pre_len = pre_len
        # self.pred_mode = pred_mode
        self.enc_step = enc_step
        self.feature_data = get_time_f_feature(
            filepath=file_path, filename=file_name, win_size=1024, step=1024)
        self.label = get_IF_label(self.feature_data[:, 3])

    def __getitem__(self, index):
        index = index * self.enc_step
        return self.feature_data[index:index + self.enc_len], self.label[index + self.enc_len + self.pre_len]



    def __len__(self):
        return (len(self.data) - self.enc_len) // self.enc_step - 1
    


def get_loader(file_path, file_name,batch_size,shuffle,win_size=1024, step=1024,enc_step = 20,enc_len=20,pre_len=1, pred_mode = 'single',fit_mode = 'train'):
        
        if fit_mode == 'train':
            shuffle = True
        else:
            shuffle = False
        data = dataset(file_path, file_name, win_size, step, enc_step, enc_len, pre_len, pred_mode)

        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        return dataloader




# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x


# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean