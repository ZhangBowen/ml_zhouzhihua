#!/usr/bin/env python3
#coding=u8

import os
import subprocess
import sys
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

import linear_discriminant_analysis

data_dir = os.environ['d_data']

if __name__ == '__main__':
    data = np.loadtxt(data_dir + '/3.0_alpha')
    p_set = data[data[:,0] == 0]
    p_set = p_set[:, 1:]
    np_set = data[data[:,0] == 1]
    np_set = np_set[:, 1:]
    theta = linear_discriminant_analysis.linear_discriminant_analysis(p_set, np_set)
    print(theta)
