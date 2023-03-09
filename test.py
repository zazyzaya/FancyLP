from types import SimpleNamespace

from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.datasets import CoraFull

from models import CattedLayerLP
from util import tr_va_te

PYG_DATA = '/home/ead/iking5/data/pyg/'

def train(model, data, hp):
    pass 

@torch.no_grad()
def eval(model, data, hp):
    pass 

def main(hp):
    data = CoraFull(PYG_DATA).data 
    tr,va,te = tr_va_te(data.edge_index.size(1))
    data.tr_mask = tr; data.va_mask = va; data.te_mask = te 

    model = CattedLayerLP(data.x.size(1), hp.hidden, hp.layers)

    train(model, data, hp)
    return eval(model, data, hp)