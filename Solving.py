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

from logger import _logger
from tffeature import *

        


class EarlyStopping:
    def __init__(self,logger, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger


    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.debug(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))
        self.val_loss_min = val_loss


class Solving(object):
    def __init__(self,args):

        #!  build model

        # self.mode = MLPMixer(args.image_size,args.patch_size,args.hidden_dim,args.depth,dropout=args.dropout,num_classes=args.num_classes)
       
        #! 设置优化器
        self.opt = torch.optim.Adam(self.mode.parameters(),lr=args.learning_rate)
        #! 设置Loss函数
        self.lossfunc = nn.MSELoss()
        
        # 赋值参数
        self.args = args
        # 记录本地时间
        local_time = time.strftime("%Y年%m月%d日 %H时%M分%S秒")
        # 查看路径是否存在
        if os.path.isdir('args.save_dir'+ f'{local_time}') is True:
            # 设置log路径
            self.log = _logger(args.save_dir + f'{local_time}/log')
        else:
            # 如果路径不存在，添加路径中的文件
            os.mkdir(args.save_dir + f'{local_time}')
            # 设置log路径
            self.log = _logger(args.save_dir + f'{local_time}/log')
        #! 设置保存路径 {参数中的保存路径 + 本地时间/}
        self.save_dir = args.save_dir + f'{local_time}/'

        # 判断cuda是否可用
        if torch.cuda.is_available():
            # 将model放在cuda上
            self.mode.cuda()
        
        #   训练数据的路径
        #   file_path = './35Hz12kN/Bearing1_1/'
        #   file_name = glob.glob(file_path+'*.csv')
        self.train_dataloader = get_loader(file_name= glob.glob(file_path+'*.csv'),file_path= './35Hz12kN/Bearing1_1/' ,mode='train',batch_size=args.batch_size)

        #    验证数据的路径
        #   file_path = './35Hz12kN/Bearing1_2/'
        #   file_name = glob.glob(file_path+'*.csv')
        self.val_dataloader = get_loader(file_name= glob.glob(file_path+'*.csv'),file_path='./35Hz12kN/Bearing1_2/',mode='val',batch_size=args.batch_size)

        #    测试数据的路径
        #    file_path = './35Hz12kN/Bearing1_3/'
        #    file_name = glob.glob(file_path+'*.csv')
        self.test_dataloader = get_loader(file_name= glob.glob(file_path+'*.csv') ,file_path='./35Hz12kN/Bearing1_3/',mode='test',batch_size=args.batch_size)
        

        #   设置早停
        self.earlystopping = EarlyStopping(logger=self.log)

        #   画图函数 
    def vis(self,train_iter_loss,val_loss):
        '''

        train_iter_loss : 训练的迭代loss

        val_loss        : 验证机的loss
        '''
        # print(val_acc)
        _,ax = plt.subplots(2,1,figsize=(20,15))
    
        
        sns.lineplot(data=train_iter_loss,ax=ax[0])
        ax[0].set_xlabel('num_iter')
        ax[0].set_ylabel('train_iter_loss')
        ax[0].set_xlim(0,train_iter_loss.shape[0])


        sns.lineplot(data=val_loss,ax=ax[1])
        ax[1].set_xlabel('num_epoch')
        ax[1].set_ylabel('val_loss')
        ax[1].set_xlim(0,val_loss.shape[0])
        plt.savefig(f'{self.save_dir}'+ 'vis.pdf',dpi=800)


    #! 验证函数  
    def val(self):
        # 设置储存验证loss的列表
        val_loss = []
        # 开启模型验证模式，其dropout 等不起作用
        self.mode.eval()
        #  循环数据：从val_dataloader取出验证集数据（data,label）
        for data, label in self.val_dataloader:
            # 将数据放在cuda上
            data = data.to(self.args.device)
            # 将标签放在cuda上
            label = label.to(self.args.device)
            # 将验证数据送入模型进行推理 得到结果 output
            output = self.mode(data)
            # 使用 loss函数 计算损失 
            loss = self.lossfunc(output,label)
            # 将Loss放入存储列表
            val_loss.append(loss.item())


        # 返回val_loss 的平均值
        return np.average(val_loss)


    #!  训练函数
    def train(self):
        # 设置储存训练loss的列表
        train_iter_loss = []

        # 设置储存验证loss的列表
        val_loss_list = []

        # 记录更新迭代的次数
        iter = 0

        #! 输出开始训练
        self.log.debug('='*10 +'Train' + '='*10)
        # self.args.epoch 为训练轮数 
        for i in range(self.args.epoch):
            
            # 初始化训练loss的值 并为后续计算平均训练loss做准备  
            train_loss = 0
            # 开启模型训练模式
            self.mode.train()
            # 在dataloader中取出数据
            for data , label in self.train_dataloader:
                # 记录更次迭代次数
                iter += 1
                # 清空梯度
                self.opt.zero_grad() 
                # 数据放在cuda
                data = data.to(self.args.device) 
                # 数据放在cuda
                label = label.to(self.args.device)
                # 数据送入模型
                output = self.mode(data)
                # 计算loss
                iter_loss = self.lossfunc(output,label)  
                # loss函数输出的是这个batch的平均值 * batch中的数据个数  = 迭代的总loss
                train_loss += self.lossfunc(output,label).item() * data.shape[0]
                # 将本次batch的Loss 平均值保存带列表
                train_iter_loss.append(iter_loss.item())
                # 梯度反向传播
                iter_loss.backward()
                # 优化器动作 更新网络参数
                self.opt.step()
            # 调用验证函数
            val_loss = self.val()
            # 验证集损失进行保存
            val_loss_list.append(val_loss)

            # 每一个epoch 输出一个综合信息
            self.log.debug(f'eopch:{i + 1} train_loss: {train_loss/len(self.train_dataloader.sampler):.2f}  val_loss:{val_loss}')
            # 早停机制 设置模型的额保存地址 
            # 设置早停的判断指标 val_loss
            self.earlystopping(val_loss ,self.mode, self.save_dir)
            # 判断是否早停
            if self.earlystopping.early_stop:
                # 输出早停信息！
                print("Early stopping")
                # 保存训练迭代损失
                np.save(f'{self.save_dir}' + 'train_iter_loss',np.array(train_iter_loss))
                # 保存验证损失
                np.save(f'{self.save_dir}' + 'val_loss',np.array(val_loss_list))
                # 跳出循环
                break
        # 如果没有早停则自然保存训练损失       
        np.save(f'{self.save_dir}' + 'train_iter_loss',np.array(train_iter_loss))
        # 如果没有早停则自然保存验证损失  
        np.save(f'{self.save_dir}' + 'val_loss',np.array(val_loss_list))
        
    #! 测试函数
    def test(self):
        # 加载模型参数
        self.mode.load_state_dict(
            torch.load(
                os.path.join(self.save_dir + '_checkpoint.pth')))
        # 开启模型测试模式
        self.mode.eval()
        # 保存预测的值
        pred_label = []
        # 保存真实的值
        true_label = []
        test_loss_list = []
        for data ,label in self.test_dataloader:
            # 放到cuda上
            data = data.to(self.args.device)
            # 输入模型
            output = self.mode(data)
            test_loss  = self.lossfunc(output.cpu(),label)
            test_loss_list.append(test_loss.item())



           # 保存预测
            pred_label.extend(list(np.array(output.cpu().detach(),axis=-1).reshape(-1)))
            # 保存真实label
            true_label.extend(list(np.array(label).reshape(-1)))


        self.log.debug(f'Test_loss : {np.average(np.array(test_loss_list)):.2f}')
        # 读取数据
        train_iter_loss = np.load(f'{self.save_dir}' + 'train_iter_loss.npy')
        val_loss = np.load(f'{self.save_dir}' + 'val_loss.npy')
        # 画图
        self.vis(train_iter_loss,val_loss)
