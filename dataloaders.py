import os 
import json 

from scipy.io import loadmat
import torch 
from torch_geometric.data import Data 
from torch_geometric.datasets import CoraFull, Planetoid, WebKB

PYG_DATA = '/home/ead/iking5/data/pyg/'
PYG_TXT_TO_DATA = {
    'cora': (Planetoid, (PYG_DATA, 'Cora')),
    'citeseer': (Planetoid, (PYG_DATA, 'CiteSeer')),
    'pubmed': (Planetoid, (PYG_DATA, 'PubMed')),
    'cornell': (WebKB, (PYG_DATA, 'Cornell')),
    'texas': (WebKB, (PYG_DATA, 'Texas')),
    'wisconsin': (WebKB, (PYG_DATA, 'Wisconsin'))
}
SCIPY_DATA = [
    'Cele','Ecoli','NS','PB','Power','Router','USAir','Yeast'
]

def load_dataset(name):
    if name in PYG_TXT_TO_DATA:
        return load_pyg(name)
    elif name in SCIPY_DATA:
        return load_scipy_mat(name)
    elif name == 'chameleon':
        return load_chameleon()
    else:
        raise ValueError(f"I don't have dataset: {name}")

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

def load_pyg(name):
    fn,arg = PYG_TXT_TO_DATA[name]
    return fn(*arg).data

def load_scipy_mat(name):
    read_dir = 'data/scipy_mats/'
    write_dir = 'data/pyg_mats/'
    out_f = write_dir+name+'.pt'

    if os.path.exists(out_f):
        return torch.load(out_f)
    
    mat = loadmat(read_dir+name+'.mat')['net']
    fmt = mat.format
    
    dst = torch.from_numpy(mat.indices)
    src = torch.zeros(dst.size(0), dtype=torch.long)
    
    ptr = mat.indptr
    num_nodes = ptr.shape[0]-1
    for i in range(num_nodes):
        src[ptr[i]:ptr[i+1]] = i 

    if fmt == 'csc':
        print("Using idx ptr as dst row")
        edge_index = torch.stack([src,dst])
    elif fmt == 'csr':
        print("Using idx ptr as src row")
        edge_index = torch.stack([dst,src])
    else:
        raise TypeError("What the heck format is %s" % fmt)
    
    g = Data(x=torch.eye(num_nodes), edge_index=edge_index)
    torch.save(g, out_f)

    return g 

