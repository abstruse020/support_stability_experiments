import sys
import yaml
import json
from types import SimpleNamespace

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