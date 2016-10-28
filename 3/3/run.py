#!/Users/zhangbowen/bin/python
#coding=u8

import os
import subprocess
import numpy as np
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

import logistic_regression

data_dir = os.environ['d_data']

if __name__ == '__main__':
    data = np.loadtxt(data_dir + '/3.0_alpha')
    train_y = data[:,0]
    train_x = np.delete(data, 0, 1)
    theta = logistic_regression.logistic_regression(train_x, train_y)
    print(theta)
