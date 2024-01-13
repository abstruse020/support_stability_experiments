import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys
import yaml
import matplotlib.pyplot as plt

from main_exp_subroutine import prob_zS_givenR, p_zS_givenR_for_m
from dataset import get_dataset, show_some_data
from utils import get_config, save_config
from model import get_model
from tqdm import  tqdm

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

## Exp details
config = get_config()

exp_name = config.exp_name
exp_dir = 'results/' + exp_name + '/'

if os.path.isdir(exp_dir):
    print(f'Folder already exists, want to override files of \'{exp_dir}\'? (y/n)')
    response = input()
    if response == 'y':
        print('#'*20,'Continuing...', '#'*20, '\n\n')
    elif response == 'n':
        print('#'*20,'Exitting...', '#'*20, '\n\n')
        exit()
    else:
        print('#'*20,'Invalid Response, Exitting...', '#'*20, '\n\n')
        exit()
else:
    os.mkdir(exp_dir)
    print('#'*20,f'New folder created: {exp_dir} ', '#'*20)


train_file = exp_dir + config.train_file
column_names = ['r_id', 'm', 's_id', 'epoch', 'train_loss', 'train_acc']
with open(train_file, 'w') as f:
    contents = ','.join(column_names) + '\n'
    f.write(contents)

prob_file = exp_dir + config.prob_file
column_names = ['r_id', 'm', 's_id', 'z_id', 'loss', 'label']
with open(prob_file, 'w') as f:
    contents = ','.join(column_names) + '\n'
    f.write(contents)

config_file = exp_dir + 'config.yaml'
save_config(config, path= config_file)

## Perform multiple experiments
def prob_zSR(config, device, prob_file, train_file):
    global_seed = config.seed
    torch.manual_seed(global_seed)
    # torch.seed_everything(global_seed)
    np.random.seed(global_seed)

    ## Setting the data distribution
    data_dist = get_dataset(config.data, device=device)
    
    ## Initialize
    model_name = config.model.name
    num_features = config.data.num_features
    num_classes = config.data.num_classes
    layers = config.model.layers
    

    r_times = config.r_times
    for r_id in tqdm(range(r_times)):
        # print(f'Running for :{itr}th iteration')
        
        ## setting param to initializa model with same values
        model_temp = get_model(model_name, num_features, num_classes, layers, local_seed=None)
        params = model_temp.state_dict()

        prob_zS_givenR(config, data_dist, r_id, params, device, prob_file, train_file)
        
    return

### Uncomment for exp 1, no increas in m 
# prob_zSR(config, device, prob_file, train_file)

def prob_zSR_for_m(config, device, prob_file, train_file):
    global_seed = config.seed
    torch.manual_seed(global_seed)
    # torch.seed_everything(global_seed)
    np.random.seed(global_seed)

    ## Setting the data distribution
    data_dist = get_dataset(config.data, device=device)
    
    ## Initialize
    model_name = config.model.name
    num_features = config.data.num_features
    num_classes = config.data.num_classes
    layers = config.model.layers
    train_sets_size = config.data.train_sets_size

    r_times = config.r_times
    for r_id in tqdm(range(r_times), desc='over random init'):
        # print(f'Running for :{itr}th iteration')
        
        ## setting param to initializa model with same values
        model_temp = get_model(model_name, num_features, num_classes, layers, local_seed=None)
        params = model_temp.state_dict()

        p_zS_givenR_for_m(config, data_dist, r_id, params, train_sets_size, device, prob_file, train_file)
        
    return

### IMP: Uncomment this for exp2, as train_set size increases i.e., m increases
prob_zSR_for_m(config, device, prob_file, train_file)