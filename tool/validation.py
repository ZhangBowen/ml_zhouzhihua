#coding=u8

import numpy as np
import logistic_regression

def k_fold(x_set, y_set, k):
    print('start k_fold validation k = {}'.format(k))
    total_sn = x_set.shape[0]
    #k组数据,每组数据所包含的样本数
    sn_group = np.array([total_sn // k] * k)
    sn_group[:total_sn % k] += 1
    #快速分割索引
    fast_index = [0] + [sn_group[:i + 1].sum() for i in range(k)]

    #shuffle样本,直接取连续区间就可以达到随机效果
    tmp_set = np.hstack((x_set, y_set.reshape(total_sn, 1)))
    np.random.shuffle(tmp_set)
    x_set = tmp_set[:,:-1]
    y_set = tmp_set[:,-1]

    error_set = []
    for i in range(k):
        print('round {} start'.format(i))
        train_x_set = np.delete(x_set, slice(fast_index[i], fast_index[i + 1]), axis = 0)
        test_x_set = x_set[fast_index[i] : fast_index[i + 1]]

        train_y_set = np.delete(y_set, slice(fast_index[i], fast_index[i + 1]), axis = 0)
        test_y_set = y_set[fast_index[i] : fast_index[i + 1]]

        theta = logistic_regression.logistic_regression(train_x_set, train_y_set)
        error_ratio = logistic_regression.error_ratio(theta, test_x_set, test_y_set)
        print('round {} done. error ratio is {}'.format(i, error_ratio))
        error_set.append(error_ratio)
    return np.array(error_set)

def LOO(x_set, y_set):
    '''Leave One Out
    留一法'''
    #当做n_sample折交叉验证即可
    return k_fold(x_set, y_set, y_set.shape[0])
