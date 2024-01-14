import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import pdb

import matplotlib.pyplot as plt


def get_dataset(data_config, device = 'cpu'):
    dataset_name = data_config.name
    
    if dataset_name == 'blob1':
        return make_blob(data_config, device=device)
    elif dataset_name == 'gaussian_v1':
        return make_gaussian_v1(data_config, device=device)
    elif dataset_name == 'mnist':
        return make_mnist(data_config, device)
    elif dataset_name == 'mydataset1':
        return make_mydataset1(data_config, device=device)
    else:
        raise('Dataset name not found')
    return

def make_mnist(data_config, device):
    training_data = datasets.MNIST(
        root='data/mnist',
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root='data/mnist',
        train=False,
        download=True,
        transform=ToTensor()
    )


    filtered_data = filter_classes(training_data, num_classes=data_config.num_classes)
    if data_config.randomize:
        filtered_data = randomise_label(filtered_data)

    return filtered_data

def make_blob(data_config, device):
    raise Exception('Dataset Not implemented')

def make_mydataset1(data_config, device):
    pass

def make_gaussian_v1():
    pass


def to_dataloader(data, batch_size, shuffle = True):
    if shuffle and False:
        print('!!'*20)
        print('Shuffle is set to True !')
        print('!!'*20)

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def filter_classes(data, num_classes = 2):
    filtered_data = []
    for img, label in data:
        for cls in range(num_classes):
            if cls == label:
                img = img.squeeze()
                filtered_data.append([img, label])
    return filtered_data

def randomise_label(data):
    # print('randomizing label')
    for i in range(len(data)):
        lable = np.random.randint(0,2)
        data[i][1] = lable
    return data

def pick_train_set(data_dist, train_size, test_idx, random_seed = None, replace=True):
    data_dist_size = len(data_dist)
    if train_size + len(test_idx) > data_dist_size:
        raise Exception('Train + test size too large', f'{train_size} + {len(test_idx)} > {data_dist_size}')
    
    train_indexes = list(set(range(data_dist_size)) - set(test_idx))
    if random_seed is None:
        picked_index = np.random.choice(train_indexes, train_size, replace=replace)
    else:
        rnp = np.random.RandomState(random_seed)
        picked_index = rnp.choice(train_indexes, train_size, replace=replace)

    train_set = list(map(lambda i: data_dist[i], picked_index))
    return train_set

## Here default value of replace is set to False as we don't want to pick a point from train set
def pick_train_test_set(data_dist, train_size, z_times, random_seed = None, replace = False):
    if train_size + z_times > len(data_dist):
        raise Exception('Train + test size too large for dataset')
    
    if random_seed is None:
        picked_index = np.random.choice(range(len(data_dist)), train_size + z_times, replace=replace)
    else:
        rnp = np.random.RandomState(random_seed)
        picked_index = rnp.choice(range(len(data_dist)), train_size + z_times, replace=replace)
    
    train_set = list(map(lambda i: data_dist[i], picked_index[:train_size]))
    test_set = list(map(lambda i: data_dist[i], picked_index[train_size:]))
    return train_set, test_set

def pick_test_idx(data_dist_size, test_size, random_seed = None):

    if test_size > data_dist_size:
        raise Exception('Test Size too large')
    
    if random_seed is None:
        test_idx = np.random.choice(range(data_dist_size), test_size, replace=False)
    else:
        rnp = np.random.RandomState(random_seed)
        test_idx = rnp.choice(range(data_dist_size), test_size, replace=False)
    
    return test_idx

## Just Visualization
def show_some_data(data_dist, n_plots = (4,4), figsize=(10,10), path=None):
    figure, axis = plt.subplots(n_plots[0], n_plots[1], figsize=figsize)
    idx = 0
    for ax_row in axis:
        for ax in ax_row:
            ax.imshow(data_dist[idx][0])
            ax.set_title(f'Random label: {data_dist[idx][1]}')
            ax.set_xticks([])
            ax.set_yticks([])
            idx +=1
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    