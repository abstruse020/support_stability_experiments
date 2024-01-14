import sys
import yaml
import json
from types import SimpleNamespace

import torch
from torch.linalg import matrix_norm as mx_norm


def get_config(config_path = 'config.yaml'):
    with open(config_path, 'r') as f:
        contents = yaml.safe_load(f)
    contents = json.loads(json.dumps(contents), object_hook = lambda x: SimpleNamespace(**x))
    return contents

def config_to_dict(config):
    return {
        k: config_to_dict(v) if isinstance(v, SimpleNamespace) else v for k,v in vars(config).items()
    }

def save_config(config, path = './results/config.yaml'):
    config_dict = config_to_dict(config)
    with open(path, '+w') as f:
        yaml.dump(config_dict, f)


## Custom steep loss function
class StepLoss(torch.nn.Module):
    def __init__(self, device) -> None:
        super(StepLoss, self).__init__()

        self.weight = 6
        self.device = device
    
    def forward(self, op, target):
        B = op.shape[0]
        op = op * self.weight
        # for numerical stability
        op = op - op.max(dim = 1, keepdim=True).values
        op_exp = torch.exp(op)
        op_exp_sum = torch.sum(op_exp, dim=1, keepdim=True)
        sm_output = 1. - op_exp/op_exp_sum
        # print('probs:', sm_output)
        loss = torch.tensor([0.]).to(self.device)
        for i in range(B):
            loss += sm_output[i][target[i].item()]
        loss = loss/B
        return loss

# To get a dictionary of model gradients, frobenious norm and spectral norm
def get_model_grads(model):
    f_norm = {}
    m_norm = {}
    for idx, param in enumerate(model.parameters()):
        f_norm[idx] = param.grad.norm(p='fro')
        if len(param.grad.shape) > 1:
            m_norm[idx] = mx_norm(param.grad, ord=2)
    
    grads = [(f'fro{key}', value.item()) for key, value in f_norm.items()]
    grads += [(f'spe{key}', value.item()) for key, value in m_norm.items()]
    return dict(grads)

def get_grads_column(config):
    
    model_name = config.model.name
    num_features = config.data.num_features
    num_classes = config.data.num_classes
    layers = config.model.layers

    model = get_model(model_name, num_features, num_classes, layers)
    raise Exception('Not Implemented')
    ## This is only for fully connected networks
