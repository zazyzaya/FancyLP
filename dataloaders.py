import os 
import json 

import torch 
from torch_geometric.data import Data 

def load_chameleon():
    data_dir = 'data/chameleon/'
    out_f = data_dir + 'chameleon.pt'
    
    # Only build once
    if os.path.exists(out_f):
        return torch.load(out_f)
    
    f = open(data_dir+'edges.csv', 'r')
    f.readline() # Skip header

    # Build edge index
    src,dst = [],[] 
    line = f.readline() 
    while(line):
        s,d = line.split(',')
        src.append(int(s)); dst.append(int(d.strip()))
        line = f.readline()

    f.close()

    # Build feature matrix
    with open(data_dir+'features.json', 'r') as f:
        feat_dict = json.load(f)

    # Stored in sparse format {'id': [idx==1].nonzero()}
    feat_dim = max([max(v) for v in feat_dict.values()])+1
    n_nodes = max(max(src), max(dst))+1
    
    x = torch.zeros(n_nodes, feat_dim)
    for k,v in feat_dict.items():
        x[int(k),v] = 1.

    # Don't really care about classes for LP 
    g = Data(x=x, edge_index=torch.tensor([src,dst]))
    torch.save(g, out_f)

    return g 

def load_mat(name):
