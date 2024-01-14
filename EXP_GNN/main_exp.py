import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import sys
import yaml
import matplotlib.pyplot as plt

from main_exp_subroutine import exp_once
from dataset import get_dataset
from utils import get_config, save_config
from model import GCN

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

## Exp details
config = get_config()

exp_name = config.exp_name
exp_dir = 'results/' + exp_name + '/'

if os.path.isdir(exp_dir):
    print(f'Folder already exists, want to override files of \'{exp_dir}\'? (y/n)')
    response = input()
    if response == 'y':
        print('*'*20,'Continuing...', '*'*20)
    elif response == 'n':
        print('*'*20,'Exitting...', '*'*20)
        exit()
    else:
        print('*'*20,'Invalid Response, Exitting...', '*'*20)
        exit()
else:
    os.mkdir(exp_dir)
    print('*'*20,f'New folder created: {exp_dir} ', '*'*20)


result_file = exp_dir + config.result_file
column_names = ['random_seed', 'pp', 'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
with open(result_file, 'w') as f:
    contents = ','.join(column_names) + '\n'
    f.write(contents)

config_file = exp_dir + 'config.yaml'
save_config(config, path= config_file)

## Setting the data
data_pp = get_dataset(dataset_name = config.data.name, device=device, preprocess = True)
data_nopp = get_dataset(dataset_name = config.data.name, device=device, preprocess = False)


## Perform multiple experiments
def multiple_exp(data_pp, data_nopp, times=10):
    random_seeds = torch.randperm(times) + 100
    iter = 1
    for local_seed in random_seeds:
        print(f'Running for :{iter}')
        exp_once(config, local_seed.item(), data_pp, device, preprocessing=True, result_file=result_file)
        exp_once(config, local_seed.item(), data_nopp, device, preprocessing=False, result_file=result_file)
        iter +=1
    return

multiple_exp(data_pp, data_nopp, 10)

