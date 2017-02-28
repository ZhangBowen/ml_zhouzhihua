#!/usr/bin/env python3
#coding=u8

import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

import decision_tree

data_dir = os.environ['d_data']

if __name__ == '__main__':
    data = pd.read_csv(data_dir + '/2.0', sep = '\t')
    X_set = data.drop(['好瓜'], axis = 1)
    Y_set = np.array(data['好瓜'])
    X_train, X_test, y_train, y_test = train_test_split(X_set, Y_set, test_size = 0.4, random_state = 42)
    #print(X_test.iloc[0]['色泽'])
    #print(X_test.T)
    #for i in X_test.T:
    #    print(i)
    #print(X_test.T.to_dict().values())
    #exit()
    
    print("normal tree:")
    tree = decision_tree.CART(X_train, y_train, [])
    tree.dump_tree()
    accuracy_ratio = tree.evaluate(X_test, y_test)
    print('准确率: {:.2f}%'.format(accuracy_ratio))

    print("pre-pruning tree:")
    tree = decision_tree.Pre_pruning_DT(X_train, y_train, X_test, y_test)
    tree.dump_tree()
    accuracy_ratio = tree.evaluate(X_test, y_test)
    print('准确率: {:.2f}%'.format(accuracy_ratio))

    print("post-pruning tree:")
    tree = decision_tree.Post_pruning_DT(X_train, y_train, X_test, y_test)
    tree.dump_tree()
    accuracy_ratio = tree.evaluate(X_test, y_test)
    print('准确率: {:.2f}%'.format(accuracy_ratio))
