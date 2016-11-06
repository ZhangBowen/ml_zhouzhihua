#coding=u8

import numpy as np

def linear_discriminant_analysis(p_set, np_set):
    #计算均值向量
    p_mean = p_set.mean(axis = 0)
    np_mean = np_set.mean(axis = 0)

    #计算协方差
    p_cov = np.cov(p_set, rowvar = False)
    np_cov = np.cov(np_set , rowvar = False)
    S_w = p_cov + np_cov
    theta = np.dot(np.linalg.inv(S_w), (p_mean - np_mean))

    return theta
