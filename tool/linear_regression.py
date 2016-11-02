#coding=u8
import numpy as np

def normal_equation(train_x, train_y):
    '''(在X.transe * X可逆的情况下)一步求出闭式解'''
    num_sample = train_x.shape[0]
    #在矩阵后增加常数列
    train_x = np.c_[train_x, np.ones(num_sample)]
    tmp = np.dot(np.transpose(train_x), train_x)
    tmp = np.dot(np.linalg.inv(tmp), np.transpose(train_x))
    theta = np.dot(tmp, train_y)
    return theta
