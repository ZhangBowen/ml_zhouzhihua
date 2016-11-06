#coding=u8
import numpy as np

import common

def logistic_regression(train_x, train_y, epoch = 100, leaning_rato = 0.01, epsilon = 1e-5):
    print('start train. epoch = {} leaning_rato = {} epsilon = {}'.format(epoch, leaning_rato, epsilon))
    num_sample = train_x.shape[0]
    #在矩阵后增加常数列
    train_x = np.c_[train_x, np.ones(num_sample)]
    #初始化参数
    theta = np.zeros(train_x.shape[1])

    for i in range(epoch):
        #全量样本参与更新
        logloss = 0
        delta_theta = np.zeros(theta.shape[0])
        for j in range(num_sample):
            x_i = train_x[j]
            p = np.dot(x_i, theta)
            h = common.sigmod(p)
            loss = h - train_y[i]
            delta_theta += leaning_rato * (loss * x_i)
            logloss += common.logloss(p, train_y[i])
        delta_theta /= num_sample
        theta -= delta_theta
        logloss /= num_sample
        print('epoch[{}/{}]logloss: {}'.format(i + 1, epoch, logloss))
        if logloss < epsilon:
            print('logloss less than threshold {}. stop'.format(epsilon))
            break
    return theta

def predict(theta, test_x):
    tmp = map(common.sigmod, np.dot(test_x, theta))
    tmp = map(common.one_or_zero, tmp)
    return np.array(list(tmp))

def error_ratio(theta, test_x, test_y):
    num_sample = test_y.shape[0]
    test_x = np.c_[test_x, np.ones(num_sample)]
    predict_y = predict(theta, test_x)
    return (num_sample - (predict_y == test_y).sum()) * 1.0 / num_sample
