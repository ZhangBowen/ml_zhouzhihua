#!/usr/bin/env python3
#coding=u8

import os
import subprocess
import sys
import pandas as pd
import numpy as np

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
    data = pd.read_csv(data_dir + '/3.0')
    X_set = data.drop(['好瓜'], axis = 1)
    Y_set = np.array(data['好瓜'])
    #指定连续属性
    continuous_attrs = ['密度', '含糖率']
    tree = decision_tree.Decision_tree_C4_5(X_set, Y_set, continuous_attrs)
    tree.dump_tree()
