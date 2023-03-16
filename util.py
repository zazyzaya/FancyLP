import torch 

def tr_va_te(samples, tr=0.85,va=0.05,te=0.10):
    assert tr+va+te == 1, 'Percentage of tr+va+te must sum to 1'

    tr_idx = int(samples * tr)
    va_idx = tr_idx + int(samples * va)

    perm = torch.randperm(samples)
    tr_mask = torch.zeros(samples, dtype=torch.bool)
    va_mask = torch.zeros(samples, dtype=torch.bool)
    te_mask = torch.zeros(samples, dtype=torch.bool)

    tr_mask[perm[:tr_idx]] = True
    va_mask[perm[tr_idx:va_idx]] = True 
    te_mask[perm[va_idx:]] = True 

    return tr_mask, va_mask, te_mask 

def negative_edges(num_nodes, samples): 
    return torch.randint(0, num_nodes, (2,samples))