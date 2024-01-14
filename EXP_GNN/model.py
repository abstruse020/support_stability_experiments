import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

## TODO: need to make layers dynamic and try mostly on just 1 layer GCN

class GCN(nn.Module, layers_width = [50]):
    def __init__(self, in_features, num_class = 2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, 50)
        self.conv2 = GCNConv(50, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)