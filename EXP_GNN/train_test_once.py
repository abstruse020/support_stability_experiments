import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

from dataset import get_dataset
from utils import get_config
from model import GCN

import os
import sys
import matplotlib.pyplot as plt

global_seed = sys.argv[1]
print('Setting global seed:', global_seed)
torch.manual_seed(global_seed)
torch_geometric.seed_everything(global_seed)

# import pdb

### IMP: Remove this later
# pdb.set_trace()

## Todo
# same randdom initialization for experiments

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


## Exp details
config_file = 'config_once.yaml'
config = get_config(config_file)

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
column_names = ['random_seed', 'pp', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
with open(result_file, 'w') as f:
    contents = ','.join(column_names) + '\n'
    f.write(contents)

## Setting the data
if config.preprocess:
    print('Preprocessing in turned on :)')
else:
    print('Preprocessing is turned off')
data = get_dataset(dataset_name = config.data.name, device=device, preprocess = config.preprocess)

## Training prep
def train_once(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    acc = (output.argmax(dim=1)[data.train_mask] == data.y[data.train_mask]).sum()
    acc = int(acc)/int(data.train_mask.sum())
    return loss.item(), acc

def test_once(model, data):
    model.eval()
    output = model(data)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct)/ int(data.test_mask.sum())
    return loss.item(), acc

## Training loop

def exp_once(config, local_seed):
    epochs = config.epochs
    num_class = config.data.num_class
    num_features = config.data.num_features
    lr = config.lr
    wt_decay = config.wt_decay

    ## TODO: set random seed value
    model = GCN(num_features,num_class).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wt_decay)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epochs):
        
        train_loss, train_acc = train_once(model, optimizer, data)

        test_loss, test_acc = test_once(model, data)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f'train loss:{train_loss:.5f}, test loss:{test_loss:.5f}', end = ' ')
        print(f'train accu:{train_acc:.5f}, test accu:{test_acc:.5f}')

        with open(result_file, 'a') as f:
            contents = [local_seed, config.preprocess ,train_loss, train_acc, test_loss, test_acc]
            contents = ','.join([str(it) for it in contents]) + '\n'
            f.write(contents)

exp_once(config, global_seed)





        
