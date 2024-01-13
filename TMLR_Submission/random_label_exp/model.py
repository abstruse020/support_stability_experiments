import torch 
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, in_features, op_features, layers):
        super(FNN, self).__init__()

        self.in_features = in_features
        self.op_features = op_features
        self.layers = layers
        self.l1 = nn.ModuleList()

        for layer in layers:
            self.l1.append(nn.Linear(in_features, layer))
            in_features = layer
        self.l1.append(nn.Linear(in_features, op_features))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for layer in self.l1[:-1]:
            x = F.relu(layer(x))

        # last layer without activation
        x = self.l1[-1](x)
        
        return x


def get_model(model_name, in_features, op_features, layers=[16], local_seed = None, params = None):

    if local_seed is not None:
        torch.manual_seed(local_seed)
    
    if model_name == 'fnn':
        model = FNN(in_features, op_features, layers)
    else:
        raise Exception(f'No model with name \'{model_name}\' found')
    
    if params is not None:
        try:
            model.load_state_dict(params)
        except:
            raise Exception('Cant load params to this model')

    return model