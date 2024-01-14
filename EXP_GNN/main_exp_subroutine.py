import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import sys
import matplotlib.pyplot as plt

from dataset import get_dataset
from utils import get_config
from model import GCN

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
def exp_once(config, local_seed, data, device, preprocessing = False, result_file = 'result.csv'):

    print('Setting local seed:', local_seed)
    torch.manual_seed(local_seed)
    torch_geometric.seed_everything(local_seed)

    epochs = config.epochs
    num_class = config.data.num_class
    num_features = config.data.num_features
    wt_decay = config.optimizer.wt_decay
    lr = config.optimizer.lr

    model = GCN(num_features, num_class).to(device)

    if config.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wt_decay)
    elif config.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise Exception('Specified Optimizer not found')
    
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

        #print(f'train loss:{train_loss:.5f}, test loss:{test_loss:.5f}', end = ' ')
        #print(f'train accu:{train_acc:.5f}, test accu:{test_acc:.5f}')

        with open(result_file, 'a') as f:
            contents = [local_seed, preprocessing , epoch, train_loss, train_acc, test_loss, test_acc]
            contents = ','.join([str(it) for it in contents]) + '\n'
            f.write(contents)





        
