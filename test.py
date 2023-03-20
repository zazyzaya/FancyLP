from copy import deepcopy
import json 
import sys 
from types import SimpleNamespace

import pandas as pd 
from sklearn.metrics import roc_auc_score, average_precision_score as ap_score
import torch
from torch.optim import Adam 

from models import *
from util import tr_va_te, negative_edges
from dataloaders import load_dataset

ORIG_PARAMS = SimpleNamespace(
    hidden=32, layers=2, 
    lr=0.01, epochs=200, 
    patience=200
)

SLOW_PARAMS = SimpleNamespace(
    hidden=32, layers=2, 
    lr=5e-5, epochs=5000, 
    patience=200
)
torch.set_num_threads(16)

def train(model, data, hp):
    opt = Adam(model.parameters(), hp.lr)
    tr_edges = data.edge_index[:, data.tr_mask]
    va_edges = data.edge_index[:, data.va_mask]

    best = (0, None)
    no_progress = 0 

    for e in range(hp.epochs):
        # Train 
        model.train()
        opt.zero_grad()
        loss = model(
            data.x, tr_edges, 
            tr_edges, negative_edges(data.x.size(0), tr_edges.size(1))
        )
        loss.backward()
        opt.step() 

        # Validate 
        stats = eval(model, data, data.va_mask, hp)
        auc,ap = stats['auc'],stats['ap']

        print(
            "[%d] Loss: %0.3f, Val AUC: %0.3f, Val AP: %0.3f" % 
            (e, loss.item(), auc,ap), end=''      
        )

        val_score = ap

        # Early stopping 
        if val_score > best[0]:
            best = (val_score, deepcopy(model.state_dict()))
            no_progress = 0 
            print('*')
        else:
            print()
            no_progress += 1 
            if no_progress > hp.patience: 
                print("Early stopping!")
                break 

    model.load_state_dict(best[1])
    return model 

@torch.no_grad()
def eval(model, data, to_test, hp, verbose=False):
    model.eval() 
    tr_edges = data.edge_index[:, data.tr_mask]
    te_edges = data.edge_index[:, to_test]
    
    pos_neg = torch.cat([te_edges, negative_edges(data.x.size(0), te_edges.size(1))], dim=1)
    labels = torch.zeros(pos_neg.size(1))
    labels[:te_edges.size(1)] = 1. 

    preds = model.predict(
        data.x, tr_edges, pos_neg
    )

    ret = dict(
        auc = roc_auc_score(labels, preds),
        ap = ap_score(labels, preds)
    )
    if verbose:
        print(json.dumps(ret, indent=1))

    return ret 

def one_test(hp, ModelConstructor, dataset):
    data = load_dataset(dataset)
    tr,va,te = tr_va_te(data.edge_index.size(1))
    data.tr_mask = tr; data.va_mask = va; data.te_mask = te 

    model = ModelConstructor(data.x.size(1), hp.hidden, hp.layers)

    train(model, data, hp)
    return eval(model, data, data.te_mask, hp, verbose=True)

def main(dataset_str, slow=False):
    hp = ORIG_PARAMS if not slow else SLOW_PARAMS

    outf = f'results/{dataset_str}.txt'
    for model in [GAE, CattedLayers_Dot, GAE_HadamardMLP, CattedLayers_HadamardMLP, GAE_DeepHadamard, CattedLayers_DeepHadamard]:
        stats = pd.DataFrame([one_test(hp, model, dataset_str) for _ in range(10)])
        print(stats.mean())

        with open(outf, 'a') as f:
            f.write(model.__name__ + '\n')
            f.write(stats.mean().to_csv())
            f.write(stats.sem().to_csv())
            f.write('\n\n')


if __name__ == '__main__':
    main(sys.argv[1], slow=True)