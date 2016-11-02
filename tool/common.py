#coding=u8
import math
import numpy as np

def sigmod(x):
    #range error fix
    x = max(-700, x)
    x = min(700, x)
    return 1 / (1 + math.exp(-x))

def one_or_zero(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def logloss(positive_probability, actuality):
    epsilon = 1e-15
    positive_probability = max(positive_probability, epsilon)
    positive_probability = min(positive_probability, 1 - epsilon)
    return actuality * math.log(positive_probability) + (1 - actuality) * math.log(1 - positive_probability)

def normalize(x_set):
    '''
    特征归一化
    input:
        待归一化数组
    output:
        归一化之后的数组
        每个维度的sum值,供新数据进行的统一的归一化
    '''
    sum_set = x_set.max(axis = 0)
    x_set /= sum_set[np.newaxis, :]
    return x_set, sum_set
