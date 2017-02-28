#coding=u8

import math
import numpy as np

def get_entropy(array_object):
    '''
    计算信息熵
    '''
    res = 0
    calc_attr = []
    total_num = len(array_object)
    for attr in array_object:
        if attr in calc_attr:
            continue
        else:
            calc_attr.append(attr)
            num = len(array_object[array_object == attr])
            if num == 0:
                continue
            probability = float(num) / total_num
            res += probability * math.log2(probability)

    return -res
            
def get_Gini(array_object):
    '''
    计算基尼指数
    '''
    res = 0
    calc_attr = []
    total_num = len(array_object)
    for attr in array_object:
        if attr in calc_attr:
            continue
        else:
            calc_attr.append(attr)
            num = len(array_object[array_object == attr])
            probability = float(num) / total_num
            res += probability ** 2

    return 1 - res
    
            

def is_same_element(array_object):
    '''
    判定该数组是否只含有一种元素
    '''
    return len(set(array_object)) == 1

def is_same_dataframe(dataframe):
    '''
    判定该dataframe中的所有行的值是否相同
    '''
    for name, value in dataframe.iteritems():
        if not is_same_element(value):
            return False
    return True

def majority_in_array(array_object):
    '''
    返回数组中占比多数的值
    '''
    count = {}
    max_value = None
    max_count = 0
    for value in array_object:
        if value not in count:
            count[value] = 0
        count[value] += 1
        if count[value] > max_count:
            max_count = count[value]
            max_value = value
    return max_value
    

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
    return -actuality * math.log(positive_probability) - (1 - actuality) * math.log(1 - positive_probability)

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
