import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

def get_dataset(dataset_name, batch_size = None, device = 'cpu', preprocess = False):

    if dataset_name == 'planetoid':
        return make_planetoid(batch_size=batch_size, device=device, preprocess = preprocess)
    
    elif dataset_name == 'mydataset1':
        return make_mydataset1(device=device)
    else:
        raise('Dataset name not found')
    return

def make_planetoid(root = '.', device = 'cpu', batch_size = None, preprocess = False):
    cora_dataset = Planetoid(root='.', name='Cora')
    cora_data = cora_dataset[0]

    if preprocess:
        cora_data.x = preprocess_x_with_deg(cora_data.x, cora_data.edge_index)
    
    cora_data = cora_data.to(device)
    return cora_data

    # Trial of batching
    # cora_loader = DataLoader(cora_dataset, batch_size=batch_size, shuffle=True)
    # return cora_loader
    
    
def make_mydataset1(device = 'cpu'):
    features = torch.rand(5,3, dtype=float) # 5 nodes, 3 features
    edges = torch.randint_like(torch.zeros(2,4), 2)

    ## Making graph un directed
    edges = torch.cat([edges, torch.stack([ edges[1], edges[0]], dim = 0)], dim = 1)

    print('features:\n', features)
    print('edges:\n', edges)

    graph_data = Data(x=features, edge_index=edges)

def preprocess_x_with_deg(x, edge_index):
    vec_deg = torch.sqrt(degree(edge_index[0]) + 1) # vec_deg[i] = sqrt(d_i + 1)
    
    vectors_along_deg = torch.matmul(x.T, vec_deg) * vec_deg.unsqueeze_(dim=1)
    vectors_along_deg /= torch.norm(vec_deg)**2
    x_pp = x - vectors_along_deg

    return x_pp

def preprocess_x_with_1s(x):
    vec_one = torch.ones(x.shape[0])
    vectors_along_one = torch.matmul(x.T, vec_one) * vec_one.unsqueeze_(dim=1)
    vectors_along_one /= torch.norm(vec_one)**2
    x_pp = x - vectors_along_one

    return x_pp