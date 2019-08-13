#!/bin/python3
import numpy as np
import os
import subprocess

samples = 3
# for learning rates, 10^-5 --> 10^-2 choose random numbers between 2 and 5
# np.random.rand returns values [0, 1)
lr_exponent = 2 + np.random.rand(samples) * 3
lr = np.power(10, -lr_exponent)
resolutions = [32]
'''losses = ['l1', 'l2', 'correlation']
reductions = ['mean', 'prod', 'worst_case']
datasets = ['z_z']
supervision = ['super', 'semi', 'unsuper']'''

resolutions = [32]
losses = ['correlation']
reductions = ['mean']
datasets = ['z_z']
supervision = ['super', 'semi', 'unsuper']


run_type = 'baseline'

name = '{}'.format(run_type)
name = name.replace('_', '-')
name = name.replace('.', '-')
name = name.replace('+', '')
print('save-{}'.format(name))
# Build command with hyperparameters specificed
kcreator_cmd = [
    'kcreator',
    '-g', '1',
    '--job-name', '{}'.format(name),
    '-i', 'chaneyk/tensorflow:v1.12.0-py3',
    '-w', '/NAS/home/projects/symmetry-detection',
    '-r', '16',
    '-b', '24',
    '-g', '1',
    '--',
    'python3', 'train.py', '--log_dir', 'save/{}/'.format(name),
]

# Create yaml file
kubectl_create_cmd = ['kubectl', 'create', '-f', '{}.yaml'.format(name)]
# Run commands in shell
subprocess.call(kcreator_cmd)
# print('kcreator_cmd', kcreator_cmd)
subprocess.call(kubectl_create_cmd)
# print('kubectl_create_cmd', kubectl_create_cmd)

get_pods_cmd = ['kubectl', 'get', 'pods']
# Run commands in shell
subprocess.call(get_pods_cmd)
