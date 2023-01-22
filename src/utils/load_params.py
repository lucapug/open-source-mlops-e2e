import yaml
from box import ConfigBox 
#we read params with yaml library and to avoid to use params['base']['feat_cols'] to pick a parameter 
#we make us of config box and the same paramter is then picked with params.base.feat_cols

def load_params(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params