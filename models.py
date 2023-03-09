import torch 
from torch import nn 
from torch_geometric.nn import GCNConv

class CattedLayerLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers) -> None:
        super().__init__()

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
        targets[pos_pred.size(0)] = 1.

        return self.loss(torch.cat([pos_pred, neg_pred]), targets)
    
    def embed(self, x, ei):
        embs = []
        for gnn in self.gnns:
            x = torch.relu(gnn(x,ei))
            embs.append(x.clone())

        return torch.stack(embs, dim=0)
    
    def lp(self, z, ei, activation=True):
        dot = z[ei[0]] @ z[ei[1]].transpose(1,2)
        emb = dot.transpose(0,1)
        
        # Concat all channels into single row so its B x C*d
        emb = emb.reshape(dot.size(0), dot.size(1)*dot.size(2))
        pred = self.pred_net(emb)

        if activation:
            return self.activation(pred)
        return pred 
            
    def predict(self, x, ei, test_edges):
        zs = self.embed(x, ei)
        return self.lp(zs, test_edges)