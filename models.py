import torch 
from torch import nn 
from torch import functional as F
from torch_geometric.nn import GCNConv

class CattedLayers_HadamardMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__()
        self.args = (in_dim, hidden_dim, layers)
        self.kwargs = dict()

        self.gnns = nn.ModuleList(
            [GCNConv(in_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(layers-1)]
        )
        self.pred_net = nn.Linear(hidden_dim*layers, 1)
        self.activation = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, ei, pos, neg):
        zs = self.embed(x,ei)

        pos_pred = self.lp(zs, pos, activation=False)
        neg_pred = self.lp(zs, neg, activation=False)

        targets = torch.zeros(pos_pred.size(0)+neg_pred.size(0), 1)
        targets[:pos_pred.size(0)] = 1.

        return self.loss(torch.cat([pos_pred, neg_pred]), targets)
    
    def embed(self, x, ei):
        embs = []
        for gnn in self.gnns:
            x = torch.relu(gnn(x,ei))
            embs.append(x.clone())

        return torch.cat(embs, dim=1)
    
    def lp(self, z, ei, activation=True):
        dot = z[ei[0]] * z[ei[1]]
        pred = self.pred_net(dot)

        if activation:
            return self.activation(pred)
        return pred 
            
    def predict(self, x, ei, test_edges):
        zs = self.embed(x, ei)
        return self.lp(zs, test_edges)
    
    
class GAE_HadamardMLP(CattedLayers_HadamardMLP):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__(in_dim, hidden_dim, layers)
        self.pred_net = nn.Linear(hidden_dim, 1)
        self.hidden = hidden_dim

    def embed(self, x, ei):
        z = super().embed(x, ei)
        return z[:, -self.hidden:]
    

class Sum(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)
    
class CattedLayers_Dot(CattedLayers_HadamardMLP):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__(in_dim, hidden_dim, layers)
        self.pred_net = Sum()

class GAE(GAE_HadamardMLP):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__(in_dim, hidden_dim, layers)
        self.pred_net = Sum()

class GAE_DeepHadamard(GAE_HadamardMLP):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__(in_dim, hidden_dim, layers)
        self.pred_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

class CattedLayers_DeepHadamard(CattedLayers_HadamardMLP):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__(in_dim, hidden_dim, layers)
        self.pred_net = nn.Sequential(
            nn.Linear(hidden_dim*layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )