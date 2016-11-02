#!/usr/bin/env python3
#coding=u8

import os
import subprocess
import sys

ROOT = os.getcwd() + '/../..'

os.environ['ROOT'] = ROOT
command = ['bash', '-c', 'source {} && env'.format(ROOT + '/conf/common.conf')]

proc = subprocess.Popen(command, stdout = subprocess.PIPE)

for line in proc.stdout:
    line = line.decode('utf-8')
    (key, value) = line.strip().split('=', 1)
    os.environ[key] = value

proc.communicate()

tool_dir = os.environ['d_tool']

sys.path.append(tool_dir)

import validation
import common

data_dir = os.environ['d_data']

import numpy as np

def k_fold_LOO(x_set, y_set):
    x_set, sum_set = common.normalize(x_set)
    #十折交叉验证
    error_ratios = validation.k_fold(x_set, y_set, 10)
    print('k_fold error ratio is {:.4f}'.format(error_ratios.mean()))

    #留一法训练过程太过暴力,略过
    #error_ratios = validation.LOO(x_set, y_set)
    #print('LOO error ratio is {:.4f}'.format(error_ratios.mean()))

if __name__ == '__main__':
    #处理skin数据集
    data = np.loadtxt(data_dir + '/skin')
    data[:,-1][data[:,-1] == 2] = 0
    y_set = data[:,-1]
    x_set = data[:,:-1]
    k_fold_LOO(x_set, y_set)

    #处理climate数据集
    data = []
    with open(data_dir + '/climate_model_simulation', 'r') as f:
        #忽略首行
        f.readline()
        for line in f:
            infos = line.strip().split()
            infos = list(map(float, infos))
            data.append(infos)
    data = np.array(data)
    x_set = data[:,:-1]
    y_set = data[:,-1]
    k_fold_LOO(x_set, y_set)
