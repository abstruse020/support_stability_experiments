import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import pdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataset, pick_train_set, pick_train_test_set, to_dataloader, show_some_data
from utils import get_config, StepLoss
from model import get_model


## Training prep
def train_once(model, optimizer, data_loader, loss_fn, epoch_idx, device):
    model.train()
    correct = 0
    running_loss = 0
    total = 0
    ##!!!!!!!!!! IMP REMOVE THIS !!!!!!!!!!
    # pdb.set_trace()

    # raise Exception('First try using pdb')
    for batch in data_loader:
        img, label = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (output.argmax(dim=1) == label).sum().item()
        total += len(label)
    
    acc = correct/(total)
    loss = running_loss/len(data_loader)
    return loss, acc

# Function not in use, needs fixing first
def test_once(model, data_loader, loss_fn, device):
    model.eval()
    
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            img, label = batch[0].to(device), batch[1].to(device)
            output = model(img)
            loss = loss_fn(output, label)
            running_loss+= loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()

    acc = int(correct)/ len(data_loader)
    loss = running_loss/ len(data_loader)
    return loss, acc

def prob_z_given_SR(config, train_loader, data_dist, r_id, s_id, params,  device, prob_file, train_file = 'train_file.csv', test_set = None):

    # print('Setting local seed:', local_seed)
    # torch.manual_seed(local_seed)
    # torch.seed_everything(local_seed)

    ## Initialization 
    epochs = config.epochs
    lr = config.optimizer.lr
    layers = config.model.layers
    replace = config.data.replace
    shuffle = config.data.shuffle
    model_name = config.model.name
    num_classes = config.data.num_classes
    batch_size = config.data.batch_size
    train_size = config.data.train_size
    wt_decay = config.optimizer.wt_decay
    num_features = config.data.num_features

    ## Settting model, optimizer and loss_fn
    model = get_model(model_name, num_features, num_classes, layers, params = params).to(device)

    if config.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wt_decay)
    elif config.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = wt_decay)
    else:
        raise Exception('Specified Optimizer not found')
    
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = StepLoss(device=device)
    
    ## Overfit the model
    # print('Overfitting...')
    train_loss_list = []
    train_acc_list = []
    # test_loss_list = []
    # test_acc_list = []
    for epoch in range(epochs):
        
        train_loss, train_acc = train_once(model, optimizer, train_loader, loss_fn, epoch, device)
        #test_loss, test_acc = test_once(model, data)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        # test_loss_list.append(test_loss)
        # test_acc_list.append(test_acc)
        # if epoch%30 == 29:
        #     print(f'epoch: {epoch:<4}, loss: {train_loss:.5f}, acc: {train_acc:.3f}')
        #print(f'train accu:{train_acc:.5f}, test accu:{test_acc:.5f}')

        with open(train_file, 'a') as f:
            contents = [r_id, len(train_loader.dataset), s_id, epoch, train_loss, train_acc]
            contents = ','.join([str(it) for it in contents]) + '\n'
            f.write(contents)
        
        if train_acc >= 0.98:
            break

    ## Calculating Loss over multiple z (points from data distribution)
    if test_set is None:
        z_times = config.z_times
        z_ids = np.random.randint(0, len(data_dist), z_times)
        test_set = data_dist
    else:
        z_ids = range(0, len(test_set))
    
    model.eval()
    loss_list = []
    label_list = []
    
    ## IMP: Remove this later
    # pdb.set_trace()

    with torch.no_grad():
        for z_id in z_ids:
            x, y = test_set[z_id][0].to(device), torch.tensor(test_set[z_id][1]).to(device)
            output = model(x.unsqueeze(dim=0))
            loss = loss_fn(output, y.unsqueeze(dim=0))
            
            loss_list.append(loss.item())
            label_list.append(y.item())
    
    ## Writing to Prob file
    contents = ''
    for loss, label, z_id in zip(loss_list, label_list, z_ids):
        temp_contents = [r_id, len(train_loader.dataset), s_id, z_id, loss, label]
        temp_contents = ','.join([str(it) for it in temp_contents]) + '\n'
        contents += temp_contents
    with open(prob_file,'+a') as f:
        f.write(contents)

def prob_zS_givenR(config, data_dist, r_id, params, device, prob_file, train_file):

    ## Initialize
    train_size = config.data.train_size
    shuffle = config.data.shuffle
    batch_size = config.data.batch_size
    replace = config.data.replace

    ## Randomly pick trian set S and call prob_z_given_SR
    s_times = config.s_times
    for s_id in range(s_times):
        train_set = pick_train_set(data_dist, train_size, replace=replace)
        # show_some_data(train_set, path='temp_img.png')

        train_loader = to_dataloader(train_set, batch_size, shuffle)
        # dist_loader = to_dataloader(data_dist, batch_size = 64, shuffle=False)

        prob_z_given_SR(config, train_loader, data_dist, r_id, s_id, params, device, prob_file, train_file)

    return

def p_zS_givenR_for_m(config, data_dist, r_id, params, train_sets_size, device = 'cpu', prob_file='prob_file.csv', train_file = 'train_file.csv'):

    ## Initialize
    # train_size = config.data.train_size
    shuffle = config.data.shuffle
    batch_size = config.data.batch_size
    replace = config.data.replace
    z_times = config.z_times

    ## Set train_set size m for multiple iterations
    for m in tqdm(train_sets_size, desc=f'loop for m'):

        ## Randomly pick trian set S and call prob_z_given_SR
        s_times = config.s_times
        for s_id in range(s_times):
            train_set, test_set = pick_train_test_set(data_dist, m, z_times, replace=replace)
            # show_some_data(train_set, path='temp_img.png')

            train_loader = to_dataloader(train_set, batch_size, shuffle)
            # dist_loader = to_dataloader(data_dist, batch_size = 64, shuffle=False)

            prob_z_given_SR(config, train_loader, data_dist, r_id, s_id, params, device, prob_file, train_file, test_set=test_set)

    return

