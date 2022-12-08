import os.path as osp
from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/data/xxy/GraphCL-master/0utils')
import multiple_embedding


class Encoder(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Encoder, self).__init__()

        self.in_dim = num_features
        self.hidden_dim = args.hidden_dim
        self.num_gc_layers = args.num_gc_layers
        self.pool = args.pool
        self.start = args.start

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.add_global_group = args.add_global_group

        for i in range(self.num_gc_layers):

            if i:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(),
                                Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.in_dim, self.hidden_dim), ReLU(),
                                Linear(self.hidden_dim, self.hidden_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(self.hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)
            in_emb_dim = (self.num_gc_layers - self.start) * self.hidden_dim
            if args.grouping_layer == 'mul_linear':
                self.grouping = multiple_embedding.MLGrouping(args, in_emb_dim)
            else:
                self.grouping = multiple_embedding.QGrouping(args, in_emb_dim)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x_group = self.grouping(torch.cat(xs[self.start:], dim=-1), batch)
        if isinstance(x_group, Tuple): 
            x_group, _ = x_group
        if self.pool == 'add':
            x_pool = [global_add_pool(x, batch) for x in xs]
        elif self.pool == 'mean':
            x_pool = [global_mean_pool(x, batch) for x in xs]
        else:
            x_pool = [global_add_pool(x, batch) for x in xs]
        # x_pool = torch.cat(x_pool[self.start:], 1)
        x_pool = x_pool[-1]
        return x_group, x_pool

    def get_embeddings(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret, ret_g = [], []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, x_global = self.forward(x, edge_index, batch)
                x = F.normalize(x, dim=-1)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())

                # global
                x_global = F.normalize(x_global, dim=-1)
                ret_g.append(x_global.cpu().numpy())
        ret = np.concatenate(ret, 0)
        ret = np.reshape(ret, [ret.shape[0], -1])

        # global
        ret_g = np.concatenate(ret_g, 0)
        # if self.add_global_group:
        ret = np.concatenate([ret, ret_g], 1)
        y = np.concatenate(y, 0)
        return ret, y

    def group_correlation(self, fea):
        fea = np.reshape(fea, [fea.shape[0], self.k, -1])
        cor = np.eye(self.k, dtype=np.float32)
        for i in range(self.k):
            for j in range(i+1, self.k):
                this_s = np.sum(fea[:, i, :] * fea[:, j, :], axis=1)
                cor[i, j] = np.mean(this_s)
                cor[j, i] = np.mean(this_s)
        return cor

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y

