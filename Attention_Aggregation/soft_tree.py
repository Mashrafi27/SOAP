# soft_tree.py
import torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------------------------------------
# 1. Soft, differentiable decision tree (one per ensemble slot)
# ------------------------------------------------------------
class SoftDecisionTree(nn.Module):
    def __init__(self, in_dim, depth=6, bin_act=torch.sigmoid):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.n_leaf = 2 ** depth
        self.bin_act = bin_act

        # feature selectors and biases -> [depth, in_dim] & [depth]
        self.selector = nn.Parameter(torch.randn(depth, in_dim))
        self.bias     = nn.Parameter(torch.zeros(depth))

        # leaf responses -> [n_leaf, 1]
        self.leaf_val = nn.Parameter(torch.zeros(self.n_leaf, 1))

        # pre-compute binary codes for all leaves  (shape [n_leaf, depth])
        code = torch.stack(
            [torch.tensor([(i >> d) & 1 for d in range(depth)])
             for i in range(self.n_leaf)]
        ).float()                # 0/1 indicators
        self.register_buffer("leaf_code", code)

    def forward(self, x):        # x : [B, in_dim]
        # routing probabilities for each internal node
        logits = F.linear(x, self.selector, self.bias)      # [B, depth]
        prob   = self.bin_act(logits)                       # [B, depth]

        # compute prob of reaching every leaf (batch-wise)
        # leaf_code == 1 -> keep prob, ==0 -> (1-prob)
        prob_expanded = prob.unsqueeze(1).expand(-1, self.n_leaf, -1)  # [B, L, depth]
        leaf_prob = torch.where(self.leaf_code == 1, prob_expanded, 1 - prob_expanded)
        leaf_prob = leaf_prob.prod(-1)                      # [B, L]

        # expected leaf value
        y_hat = leaf_prob @ self.leaf_val                   # [B, 1]
        return y_hat, leaf_prob                             # (prediction, routing)
        

# ------------------------------------------------------------
# 2.  Ensemble of soft trees  (NODE-like)
# ------------------------------------------------------------
class SoftTreeEnsemble(nn.Module):
    def __init__(self, in_dim, n_trees=64, depth=6):
        super().__init__()
        self.trees = nn.ModuleList(
            [SoftDecisionTree(in_dim, depth) for _ in range(n_trees)]
        )
        # learnable weight per tree  (like η in ChemXTree)
        self.eta = nn.Parameter(torch.ones(n_trees))

        # global bias (T0) for regression
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):                    # x : [B, in_dim]
        preds = []
        for tree in self.trees:              # list of [B,1]
            p, _ = tree(x)
            preds.append(p)
        preds = torch.cat(preds, dim=1)      # [B, n_trees]

        # weighted sum over trees
        y_hat = (preds * self.eta).sum(dim=1, keepdim=True) + self.bias
        return y_hat.squeeze(1)              # [B]
