import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from termcolor import cprint

class QGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim, key_dim=100):
        super(QGrouping, self).__init__()
        cprint(f'Initail a query grouping layer', 'green')
        embedding_dim = args.embedding_dim
        self.k = args.num_group

        self.key_dim = key_dim
        self.val_dim = embedding_dim // self.k
        args.dim_per_group = embedding_dim // self.k
        self.att_norm = args.att_norm
        self.bias = args.bias

        if self.bias == 'true':
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1)
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1)
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1)
        else:
            self.w_k = torch.nn.Conv2d(in_emb_dim, key_dim, kernel_size=1, bias=False)
            self.w_q = torch.nn.Conv2d(key_dim, self.k, 1, bias=False)
            self.w_v = torch.nn.Conv2d(in_emb_dim, self.val_dim, kernel_size=1, bias=False)

    def forward(self, x, batch):
        key = self.w_k(x.unsqueeze(2).unsqueeze(3))
        val = self.w_v(x.unsqueeze(2).unsqueeze(3))
        val = val.squeeze()
        weights = self.w_q(key).reshape((-1, self.k))
        norm_w = []
        embs = []
        for b in torch.unique(batch):
            if self.att_norm == 'softmax':
                this_w = F.softmax(weights[batch == b, :], dim=0)  # num_nodes * k
            elif self.att_norm == 'sigmoid':
                a = torch.sigmoid(weights[batch == b, :])
                num_nodes = sum(batch == b)
                this_w = a / torch.sqrt(num_nodes.float())
            else:
                raise ValueError
            this_val = val[batch == b, :]  # num_nodes * dim
            this_embs = torch.matmul(this_w.T, this_val)
            embs.append(this_embs.unsqueeze(0))
            norm_w.append(this_w)
        return torch.cat(embs, dim=0), norm_w

class MLGrouping(torch.nn.Module):
    def __init__(self, args, in_emb_dim) -> None:
        super(MLGrouping, self).__init__()
        cprint(f'Initail a mul_linear grouping layer', 'green')
        self.pool = args.pool
        self.k = args.num_group
        self.dim_per_group = args.embedding_dim // self.k
        self.ML_ops = torch.nn.ModuleList()
        for i in range(self.k):
            self.ML_ops.append(torch.nn.Linear(in_emb_dim, self.dim_per_group))
        
    def forward(self, x, batch):
        out = []
        for i in range(self.k):
            fea = self.ML_ops[i](x)
            if self.pool == 'mean':
                fea = global_mean_pool(fea, batch)
            else:
                fea = global_add_pool(fea, batch)
            out.append(fea.unsqueeze(1))
        return torch.cat(out, dim=1)

